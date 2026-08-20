from __future__ import annotations

import ast
import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

logging.getLogger("realtime").setLevel(logging.INFO)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import env
import inspect_api
import oapi
import otel
from aligner import Qwen3ForcedAligner
from diarization import (
    DiarConfig,
    DiarSegment,
    Diarizer,
    EmbeddingModel,
    SegmentationModel,
    cosine_sim,
)
from oapi import kind, task
from omni.gemma import Gemma4Wrapper
from omni.qwen3 import (
    DEFAULT_SPEAKER as OMNI_DEFAULT_SPEAKER,
    SUPPORTED_SPEAKERS as OMNI_SPEAKERS,
    Qwen3OmniWrapper,
)
from tts.kokoro import (
    DEFAULT_LANGUAGE as KOKORO_DEFAULT_LANGUAGE,
    DEFAULT_VOICE as KOKORO_DEFAULT_VOICE,
    KOKORO_HF_REPO,
    KOKORO_LANGUAGES,
    KOKORO_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SPEED_MAX,
    SPEED_MIN,
    KokoroTTS,
    f32_to_s16le,
    is_openai_voice_alias,
    normalize_for_tts,
    strip_emojis,
    strip_markdown_emphasis,
)
from tts.qwen3 import Qwen3TTSModel
from tts.qwen3.inference.streaming import chunked_decode_pcm

DEFAULT_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_DEVICE_HINT = "auto"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_BATCH_WINDOW_MS = 0

DEFAULT_TTS_MAX_NEW_TOKENS = 1024
DEFAULT_TTS_SPEED = 1.0
DEFAULT_CUSTOMVOICE_SPEAKER = "Ryan"
DEFAULT_VOICE_DESIGN_INSTRUCTION = "warm, neutral, natural speaking voice"

CHAT_DEFAULT_MAX_TOKENS = 512
CHAT_MAX_TOKENS_LIMIT = 8192
TALKER_DEFAULT_MAX_NEW_TOKENS = 4096
TALKER_MAX_NEW_TOKENS_LIMIT = 16384

MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024
BYTES_PER_MIB = 1024 * 1024

QWEN_TTS_OUTPUT_SR = 24000
PCM_AUDIO_MEDIA_TEMPLATE = "audio/L16; rate={sample_rate}; channels=1"
WAV_MEDIA_TYPE = "audio/wav"
SRT_MEDIA_TYPE = "application/x-subrip"
VTT_MEDIA_TYPE = "text/vtt"
PLAIN_TEXT_MEDIA_TYPE = "text/plain"
SSE_MEDIA_TYPE = "text/event-stream"

SPEECH_FORMAT_MIME: dict[str, str] = {
    "wav": WAV_MEDIA_TYPE,
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
}
SPEECH_FFMPEG_FORMATS = frozenset({"mp3", "flac", "opus", "aac"})

DEFAULT_TRANSLATION_PROMPT = "Translate this audio to English."
LANGUAGE_DETECTION_PROMPT_TEMPLATE = (
    "Identify the language of this audio. "
    "Reply with exactly one word from this list: {options}."
)

SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
MS_PER_SECOND = 1000

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

QWEN_TTS_GENERATE_KWARGS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "subtalker_top_p": 0.9,
    "repetition_penalty": 1.1,
}

CUSTOMVOICE_SPEAKER_LANGUAGES: dict[str, str] = {
    "Vivian": "Chinese",
    "Serena": "Chinese",
    "Uncle_Fu": "Chinese",
    "Dylan": "Chinese",
    "Eric": "Chinese",
    "Ryan": "English",
    "Aiden": "English",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}

ALIGNER_LANGUAGES = (
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Portuguese", "Russian", "Spanish",
)
ALIGNER_LANGUAGE_LOOKUP = {name.lower(): name for name in ALIGNER_LANGUAGES}

LANGUAGE_AUTO = "auto"
ENGLISH_LANGUAGE = "English"

KOKORO_TASK_NAME = "Kokoro"

def _env_csv(name: str, fallback: str) -> list[str]:
    raw = os.environ.get(name) or fallback
    return [item.strip() for item in raw.split(",") if item.strip()]

MODEL_IDS = _env_csv(
    env.QWEN3_TTS_MODELS,
    os.environ.get(env.QWEN3_TTS_MODEL) or DEFAULT_TTS_MODEL_ID,
)
DEVICE_HINT = env.read_str(env.QWEN3_TTS_DEVICE, DEFAULT_DEVICE_HINT)
DTYPE_NAME = env.read_str(env.QWEN3_TTS_DTYPE, DEFAULT_DTYPE)
HOST = env.read_str(env.QWEN3_TTS_HOST, DEFAULT_HOST)
PORT = env.read_int(env.QWEN3_TTS_PORT, DEFAULT_PORT)
BATCH_WINDOW_MS = env.read_int(env.QWEN3_TTS_BATCH_WINDOW_MS, DEFAULT_BATCH_WINDOW_MS)

OMNI_MODEL_ID = env.read_str(env.QWEN3_OMNI_MODEL, "").strip()
OMNI_DISABLE_TALKER = env.read_bool(env.QWEN3_OMNI_DISABLE_TALKER, default=False)

ALIGNER_MODEL_ID = env.read_str(env.QWEN3_ALIGNER_MODEL, DEFAULT_ALIGNER_MODEL_ID).strip()

GEMMA_MODEL_ID = env.read_str(env.GEMMA_MODEL, "").strip()

KOKORO_ENABLED = env.read_bool(env.KOKORO_ENABLE, default=False)
KOKORO_VOICES_DIR = env.read_str_or_none(env.KOKORO_VOICES_DIR)

DIAR_SEGMENTATION_MODEL_FILE = env.read_str_or_none(env.DIAR_SEGMENTATION_MODEL_FILE)
DIAR_EMBEDDING_MODEL_FILE = env.read_str_or_none(env.DIAR_EMBEDDING_MODEL_FILE)
DIAR_ENABLED = bool(DIAR_SEGMENTATION_MODEL_FILE and DIAR_EMBEDDING_MODEL_FILE)

STT_BACKEND_RAW = env.read_str(env.STT_BACKEND, "qwen3_omni").strip().lower()
STT_BACKEND_WHISPER_ALIASES = frozenset({"whisper", "ct2", "ctranslate2", "faster-whisper"})
STT_BACKEND_QWEN3_OMNI = "qwen3_omni"
STT_WHISPER_DEFAULT_DEVICE = "cpu"
STT_WHISPER_DEFAULT_COMPUTE = "default"

def _resolve_device() -> str:
    if DEVICE_HINT != "auto":
        return DEVICE_HINT
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _resolve_torch_dtype(name: str) -> torch.dtype:
    return DTYPE_BY_NAME[name]

def _resolve_task_type(model_id: str, override: str | None) -> str:
    if override:
        return override
    name = model_id.lower()
    if "voicedesign" in name or "voice-design" in name:
        return "VoiceDesign"
    if name.endswith("-base") or "12hz-1.7b-base" in name or "12hz-0.6b-base" in name:
        return "Base"
    return "CustomVoice"

REF_AUDIO_FETCH_TIMEOUT_SECONDS = 10
REF_AUDIO_MAX_BYTES = 32 * 1024 * 1024
REF_AUDIO_ALLOWED_SCHEMES = ("https",)

def _decode_ref_audio(spec: str) -> tuple[bytes, str]:
    if spec.startswith("data:") and ";base64," in spec:
        payload = spec.split(",", 1)[1]
        decoded = base64.b64decode(payload, validate=True)
        if len(decoded) > REF_AUDIO_MAX_BYTES:
            raise ValueError(
                f"ref_audio data: payload exceeds {REF_AUDIO_MAX_BYTES // (1024 * 1024)} MiB cap"
            )
        return decoded, ".wav"
    if spec.startswith("https://"):
        parsed = urlparse(spec)
        if parsed.scheme not in REF_AUDIO_ALLOWED_SCHEMES:
            raise ValueError(f"ref_audio scheme {parsed.scheme!r} not allowed; use https:// or data:")
        with urlopen(spec, timeout=REF_AUDIO_FETCH_TIMEOUT_SECONDS) as response:
            data = response.read(REF_AUDIO_MAX_BYTES + 1)
            if len(data) > REF_AUDIO_MAX_BYTES:
                raise ValueError(
                    f"ref_audio download exceeds {REF_AUDIO_MAX_BYTES // (1024 * 1024)} MiB cap"
                )
            suffix = Path(parsed.path).suffix or ".wav"
            return data, suffix
    raise ValueError(
        "ref_audio must be data:audio/...;base64,... or https://; file://, http://, "
        f"and absolute paths are blocked. Got: {spec[:60]!r}..."
    )

def _customvoice_language(speaker: str, requested_language: str) -> str:
    if requested_language != "Auto":
        return requested_language
    return CUSTOMVOICE_SPEAKER_LANGUAGES.get(speaker, "Auto")

def _voice_design_instruction(req_instructions: str, req_voice: str) -> str:
    return req_instructions or req_voice or DEFAULT_VOICE_DESIGN_INSTRUCTION

_models: dict[str, Qwen3TTSModel] = {}
_models_by_task: dict[str, Qwen3TTSModel] = {}
_device: str = "cpu"

_voice_profiles: dict[str, dict[str, Any]] = {}
_voice_profiles_lock = threading.Lock()

_batch_window_seconds = BATCH_WINDOW_MS / MS_PER_SECOND

_omni: Qwen3OmniWrapper | None = None
_omni_load_error: str | None = None
_omni_lock = threading.Lock()

_aligner: Qwen3ForcedAligner | None = None
_aligner_load_error: str | None = None

_gemma: Gemma4Wrapper | None = None
_gemma_load_error: str | None = None
_gemma_lock = threading.Lock()

_kokoro: KokoroTTS | None = None
_kokoro_load_error: str | None = None

_diarizer: Diarizer | None = None
_diarizer_load_error: str | None = None
_diarizer_lock = threading.Lock()
_diar_embedding: EmbeddingModel | None = None

_stt_backend: Any = None
_stt_backend_kind: str = STT_BACKEND_QWEN3_OMNI
_stt_backend_model_id: str | None = None
_stt_backend_load_error: str | None = None

def _load_tts_models(dtype: torch.dtype) -> None:
    for model_id in MODEL_IDS:
        model = Qwen3TTSModel.from_pretrained(
            model_id, device_map=_device, dtype=dtype, attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        _models[model_id] = model
        _models_by_task.setdefault(_resolve_task_type(model_id, None), model)

def _load_aligner_eagerly(dtype: torch.dtype) -> None:
    global _aligner, _aligner_load_error
    if not ALIGNER_MODEL_ID:
        return
    try:
        _aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL_ID, device_map=_device, dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        _aligner_load_error = f"{type(exc).__name__}: {exc}"

def _load_kokoro_eagerly() -> None:
    global _kokoro, _kokoro_load_error
    if not KOKORO_ENABLED:
        return
    try:
        _kokoro = KokoroTTS(voices_dir=KOKORO_VOICES_DIR)
    except Exception as exc:
        _kokoro_load_error = f"{type(exc).__name__}: {exc}"

def _load_diarizer_eagerly() -> None:
    global _diarizer, _diarizer_load_error, _diar_embedding
    if not DIAR_ENABLED:
        return
    try:
        seg = SegmentationModel.load(DIAR_SEGMENTATION_MODEL_FILE)
        emb = EmbeddingModel.load(DIAR_EMBEDDING_MODEL_FILE)
        _diarizer = Diarizer(seg, emb, DiarConfig.from_env())
        _diar_embedding = emb
    except Exception as exc:
        _diarizer_load_error = f"{type(exc).__name__}: {exc}"

def _load_stt_backend_eagerly() -> None:
    global _stt_backend, _stt_backend_kind, _stt_backend_model_id, _stt_backend_load_error
    if STT_BACKEND_RAW not in STT_BACKEND_WHISPER_ALIASES:
        _stt_backend_kind = STT_BACKEND_QWEN3_OMNI
        return
    speaches_models = _env_csv(env.SPEACHES_PLUS_MODELS, "")
    model_path = speaches_models[0] if speaches_models else ""
    if not model_path:
        _stt_backend_load_error = (
            f"STT_BACKEND={STT_BACKEND_RAW!r} requires SPEACHES_PLUS_MODELS to be set "
            f"to a comma-separated list with at least one entry (model path or HF id)."
        )
        import warnings
        warnings.warn(
            f"STT backend whisper requested but {_stt_backend_load_error} "
            "Falling back to qwen3_omni for /v1/audio/transcriptions.",
            stacklevel=2,
        )
        _stt_backend_kind = STT_BACKEND_QWEN3_OMNI
        return
    try:
        from stt.ct2 import Ct2WhisperBackend, Ct2WhisperConfig
        cfg = Ct2WhisperConfig(
            model_path=model_path,
            device=STT_WHISPER_DEFAULT_DEVICE,
            compute_type=STT_WHISPER_DEFAULT_COMPUTE,
        )
        _stt_backend = Ct2WhisperBackend(cfg)
        _stt_backend_kind = "whisper"
        _stt_backend_model_id = model_path
    except Exception as exc:
        _stt_backend_load_error = f"{type(exc).__name__}: {exc}"
        import warnings
        warnings.warn(
            f"STT whisper backend failed to load ({_stt_backend_load_error}); "
            "falling back to qwen3_omni for /v1/audio/transcriptions.",
            stacklevel=2,
        )
        _stt_backend = None
        _stt_backend_kind = STT_BACKEND_QWEN3_OMNI

_inspect_retention_task: Any = None

_realtime_vad_model: Any = None
_realtime_vad_load_error: str | None = None

def _load_realtime_vad_eagerly() -> None:
    global _realtime_vad_model, _realtime_vad_load_error
    path = env.read_str_or_none(env.VAD_MODEL_FILE)
    if not path:
        return
    try:
        from vad.silero import SileroVad

        _realtime_vad_model = SileroVad.load(path)
    except Exception as exc:
        _realtime_vad_load_error = f"{type(exc).__name__}: {exc}"
        import warnings
        warnings.warn(
            f"Realtime VAD model failed to load from {path!r} "
            f"({_realtime_vad_load_error}); /v1/realtime sessions will negotiate but "
            "emit no transcription events.",
            stacklevel=2,
        )
        _realtime_vad_model = None

def _build_realtime_transcribe(session: Any) -> Any:
    """Return an `async (np.ndarray) -> str` for one realtime session.

    Picks whisper-ct2 when available, otherwise Gemma. Falls back to a
    no-op (`return ""`) when no engine can handle the request -- better
    than failing the whole session.
    """
    import asyncio as _aio
    import io as _io

    import numpy as _np
    import soundfile as _sf

    def _ndarray_to_data_uri(audio: _np.ndarray, sr: int = 16000) -> str:
        buf = _io.BytesIO()
        _sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return _audio_data_uri(buf.getvalue(), ".wav")

    if _stt_backend is not None:
        backend = _stt_backend

        async def transcribe(audio: _np.ndarray) -> str:
            return await _aio.to_thread(
                lambda: backend.transcribe(audio, sample_rate=16000),
            )

        return transcribe

    if GEMMA_MODEL_ID:
        async def transcribe(audio: _np.ndarray) -> str:
            uri = _ndarray_to_data_uri(audio)
            engine = await _aio.to_thread(_load_gemma)
            return await _aio.to_thread(lambda: engine.transcribe(uri))

        return transcribe

    if OMNI_MODEL_ID:
        async def transcribe(audio: _np.ndarray) -> str:
            uri = _ndarray_to_data_uri(audio)
            engine = await _aio.to_thread(_load_omni)
            return await _aio.to_thread(lambda: engine.transcribe(uri))

        return transcribe

    async def transcribe(_audio: _np.ndarray) -> str:
        return ""

    return transcribe

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _device, _aligner, _kokoro, _diarizer, _diar_embedding, _inspect_retention_task
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        import warnings
        warnings.warn(
            f"speaches-plus-python bound to {HOST!r} with no authentication on "
            "/v1/chat/completions, /v1/audio/speech, /v1/voice-profiles, "
            "/v1/audio/transcriptions, /v1/realtime (POST SDP + WS). Anyone "
            "with network reach can submit tools, schemas, ref_audio URLs, "
            "and open realtime sessions. Bind to 127.0.0.1 or front with a "
            "reverse proxy enforcing auth.",
            stacklevel=2,
        )
    otel.init()
    _device = _resolve_device()
    dtype = _resolve_torch_dtype(DTYPE_NAME)
    _load_tts_models(dtype)
    _load_aligner_eagerly(dtype)
    _load_kokoro_eagerly()
    _load_diarizer_eagerly()
    _load_stt_backend_eagerly()
    _load_realtime_vad_eagerly()
    from realtime.transport import RealtimeContext, set_context
    import asyncio as _asyncio
    inspect_api.run_startup_cleanup()
    set_context(RealtimeContext(
        models=_realtime_models_view,
        observer_factory=inspect_api.make_observer_factory(),
        vad_model=_realtime_vad_model,
        transcribe_factory=_build_realtime_transcribe if _realtime_vad_model is not None else None,
    ))
    if inspect_api.session_dir() is not None:
        _inspect_retention_task = _asyncio.create_task(inspect_api.retention_loop())
    yield
    if _inspect_retention_task is not None:
        _inspect_retention_task.cancel()
        try:
            await _inspect_retention_task
        except BaseException:
            pass
        _inspect_retention_task = None
    inspect_api.clear_registry()
    set_context(None)
    _models.clear()
    _models_by_task.clear()
    _aligner = None
    _kokoro = None
    _diarizer = None
    _diar_embedding = None
    global _stt_backend
    if _stt_backend is not None:
        try:
            _stt_backend.close()
        except Exception:
            pass
        _stt_backend = None
    otel.shutdown()

app = FastAPI(title="speaches-plus-python", lifespan=lifespan)
app.include_router(inspect_api.router)

@app.exception_handler(HTTPException)
async def _http_exception_handler(_request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(detail, status_code=exc.status_code, headers=exc.headers)
    return JSONResponse(
        {"detail": detail}, status_code=exc.status_code, headers=exc.headers,
    )

def _load_omni() -> Qwen3OmniWrapper:
    global _omni, _omni_load_error
    if _omni is not None:
        return _omni
    if not OMNI_MODEL_ID:
        raise HTTPException(
            503,
            "Qwen3-Omni endpoints disabled. Set QWEN3_OMNI_MODEL to a HF id "
            "(e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct) to enable chat + transcription.",
        )
    with _omni_lock:
        if _omni is not None:
            return _omni
        if _omni_load_error is not None:
            raise HTTPException(503, f"Qwen3-Omni load previously failed: {_omni_load_error}")
        try:
            _omni = Qwen3OmniWrapper.from_pretrained(
                OMNI_MODEL_ID,
                dtype=_resolve_torch_dtype(DTYPE_NAME),
                device_map=_device,
                attn_implementation="eager",
                disable_talker=OMNI_DISABLE_TALKER,
            )
        except Exception as exc:
            _omni_load_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(503, f"Qwen3-Omni failed to load: {_omni_load_error}") from exc
    return _omni

def _load_gemma() -> Gemma4Wrapper:
    global _gemma, _gemma_load_error
    if _gemma is not None:
        return _gemma
    if not GEMMA_MODEL_ID:
        raise HTTPException(
            503,
            "Gemma 4 endpoints disabled. Set GEMMA_MODEL to a HF id "
            "(e.g. google/gemma-4-E4B-it for testing, google/gemma-4-31B-it "
            "for production).",
        )
    with _gemma_lock:
        if _gemma is not None:
            return _gemma
        if _gemma_load_error is not None:
            raise HTTPException(503, f"Gemma 4 load previously failed: {_gemma_load_error}")
        try:
            _gemma = Gemma4Wrapper.from_pretrained(
                GEMMA_MODEL_ID,
                dtype=_resolve_torch_dtype(DTYPE_NAME),
                device_map=_device,
            )
        except Exception as exc:
            _gemma_load_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(503, f"Gemma 4 failed to load: {_gemma_load_error}") from exc
    return _gemma

def _kokoro_or_503() -> KokoroTTS:
    if _kokoro is not None:
        return _kokoro
    if _kokoro_load_error is not None:
        raise HTTPException(503, f"Kokoro load failed at boot: {_kokoro_load_error}")
    raise HTTPException(
        503,
        "Kokoro plane disabled. Set KOKORO_ENABLE=1 (model resolves from "
        "HF_HUB_CACHE; optional KOKORO_VOICES_DIR enables full offline voice listing).",
    )

def _model_basename(model_field: str) -> str:
    return model_field.rsplit("/", 1)[-1].lower()

def _request_picks_kokoro(req: SpeechRequest) -> bool:
    if req.task_type == KOKORO_TASK_NAME:
        return True
    field = (req.model or "").strip()
    if not field:
        return False
    return _model_basename(field).startswith("kokoro")

def _request_picks_gemma(model_field: str | None) -> bool:
    if not model_field:
        return False
    if model_field == GEMMA_MODEL_ID:
        return True
    return _model_basename(model_field).startswith("gemma")

def _model_for_task(task: str) -> Qwen3TTSModel:
    model = _models_by_task.get(task)
    if model is None:
        loaded = sorted(_models_by_task)
        raise HTTPException(
            400,
            f"task_type={task!r} has no model loaded. Loaded for: {loaded}. "
            f"Set QWEN3_TTS_MODELS to a comma-separated list of HF ids covering "
            f"the task types you need.",
        )
    return model

def _clean_speech_input(text: str) -> str:
    return normalize_for_tts(strip_markdown_emphasis(strip_emojis(text)))

_FFMPEG_FORMAT_FLAGS: dict[str, list[str]] = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame"],
    "flac": ["-f", "flac"],
    "opus": ["-f", "opus", "-codec:a", "libopus"],
    "aac": ["-f", "adts", "-codec:a", "aac"],
    "wav": ["-f", "wav"],
}

def _ffmpeg_args(target_format: str, source_sr: int, target_sr: int) -> list[str]:
    flags = _FFMPEG_FORMAT_FLAGS.get(target_format)
    if flags is None:
        raise ValueError(f"unsupported ffmpeg format: {target_format!r}")
    return [
        "ffmpeg",
        "-f", "s16le", "-ar", str(source_sr), "-ac", "1", "-i", "pipe:0",
        "-ar", str(target_sr),
        *flags,
        "pipe:1", "-hide_banner", "-loglevel", "error",
    ]

def _require_ffmpeg(target_format: str) -> None:
    if shutil.which("ffmpeg") is None:
        oapi.raise_openai_error(
            503,
            f"ffmpeg not installed; response_format={target_format!r} requires ffmpeg",
            kind.SERVICE_UNAVAIL,
            param="response_format",
            code="ffmpeg_missing",
        )

def _pcm_chunks_through_ffmpeg(
    pcm_iter,
    source_sr: int,
    target_sr: int,
    target_format: str,
):
    proc = subprocess.Popen(
        _ffmpeg_args(target_format, source_sr, target_sr),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def feed():
        try:
            for chunk in pcm_iter:
                if not chunk:
                    continue
                proc.stdin.write(chunk)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            yield data
    finally:
        feeder.join(timeout=5)
        proc.wait(timeout=5)

def _encode_pcm_once(
    pcm_bytes: bytes,
    source_sr: int,
    target_sr: int,
    target_format: str,
) -> bytes:
    result = subprocess.run(
        _ffmpeg_args(target_format, source_sr, target_sr),
        input=pcm_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        oapi.raise_openai_error(
            500,
            f"ffmpeg encode failed: {stderr or 'unknown error'}",
            kind.SERVER,
            code="ffmpeg_encode_failed",
        )
    return result.stdout

def _audio_duration_seconds_or_none(contents: bytes) -> float | None:
    try:
        with sf.SoundFile(io.BytesIO(contents)) as handle:
            return float(handle.frames) / float(handle.samplerate)
    except (sf.LibsndfileError, RuntimeError, ValueError):
        return None

def _sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()

_ZERO_TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def _sse_speech_events(pcm_iter):
    for pcm in pcm_iter:
        if pcm:
            yield _sse_event({
                "type": "speech.audio.delta",
                "audio": base64.b64encode(pcm).decode("ascii"),
            })
    yield _sse_event({"type": "speech.audio.done", "token_usage": _ZERO_TOKEN_USAGE})

def _validate_speech_request(req: SpeechRequest) -> None:
    entries: list[dict[str, Any]] = []
    if not req.input:
        entries.append(oapi.missing_field(["body", "input"]))
    if not req.voice:
        entries.append(oapi.missing_field(["body", "voice"]))
    if req.sample_rate is not None and not (
        MIN_SAMPLE_RATE <= req.sample_rate <= MAX_SAMPLE_RATE
    ):
        entries.append({
            "type": "less_than_equal",
            "loc": ["body", "sample_rate"],
            "msg": (
                f"Input should be between {MIN_SAMPLE_RATE} "
                f"and {MAX_SAMPLE_RATE}"
            ),
            "input": req.sample_rate,
        })
    if entries:
        raise HTTPException(status_code=422, detail=entries)
    if not SPEED_MIN <= req.speed <= SPEED_MAX:
        oapi.raise_openai_error(
            400,
            f"speed must be between {SPEED_MIN:.1f} and "
            f"{SPEED_MAX:.1f}, got {req.speed}",
            kind.INVALID_REQUEST,
            param="speed",
            code="out_of_range",
        )

def _resolve_stream_format(req: SpeechRequest) -> str | None:
    return req.stream_format or ("audio" if req.stream else None)

class SpeechRequest(BaseModel):
    input: str
    voice: str = ""
    model: str | None = None
    response_format: Literal["wav", "pcm", "mp3", "flac", "opus", "aac"] = "wav"
    stream: bool = False
    stream_format: Literal["audio", "sse"] | None = None
    sample_rate: int | None = None
    task_type: Literal["CustomVoice", "VoiceDesign", "Base", "Kokoro"] | None = None
    language: str = "Auto"
    instructions: str = ""
    ref_audio: str | None = None
    ref_text: str = ""
    x_vector_only_mode: bool = False
    voice_profile: str | None = None
    max_new_tokens: int = DEFAULT_TTS_MAX_NEW_TOKENS
    speed: float = DEFAULT_TTS_SPEED

class VoiceProfileRequest(BaseModel):
    name: str
    ref_audio: str
    ref_text: str = ""
    x_vector_only_mode: bool = False
    model_id: str | None = None

class ChatAudioOptions(BaseModel):
    voice: str = OMNI_DEFAULT_SPEAKER
    format: Literal["wav", "pcm"] = "wav"

class ChatJsonSchemaSpec(BaseModel):
    name: str
    schema: dict[str, Any]
    strict: bool = False

class ChatResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: ChatJsonSchemaSpec | None = None

class ChatToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = Field(
        default=None, description="JSON schema for the tool's parameters",
    )

class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatToolFunction

class ChatToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    function: ChatToolFunction

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = Field(CHAT_DEFAULT_MAX_TOKENS, ge=1, le=CHAT_MAX_TOKENS_LIMIT)
    talker_max_new_tokens: int = Field(
        TALKER_DEFAULT_MAX_NEW_TOKENS, ge=1, le=TALKER_MAX_NEW_TOKENS_LIMIT,
    )
    temperature: float | None = None
    stream: bool = False
    modalities: list[Literal["text", "audio"]] = Field(default_factory=lambda: ["text"])
    audio: ChatAudioOptions | None = None
    use_audio_in_video: bool = False
    response_format: ChatResponseFormat | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["auto", "none", "required"] | ChatToolChoiceFunction | None = None

class _Batcher:
    def __init__(self, window_seconds: float):
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._pending: dict[tuple, list] = {}
        self._timers: dict[tuple, threading.Timer] = {}

    def submit(self, key: tuple, payload: dict, *, runner) -> Any:
        completion_event = threading.Event()
        slot = {
            "payload": payload,
            "result": None,
            "error": None,
            "event": completion_event,
        }
        with self._lock:
            self._pending.setdefault(key, []).append(slot)
            if key not in self._timers:
                timer = threading.Timer(self._window_seconds, self._flush, args=(key, runner))
                timer.daemon = True
                self._timers[key] = timer
                timer.start()
        completion_event.wait()
        if slot["error"] is not None:
            raise slot["error"]
        return slot["result"]

    def _flush(self, key: tuple, runner) -> None:
        with self._lock:
            slots = self._pending.pop(key, [])
            self._timers.pop(key, None)
        if not slots:
            return
        try:
            results = runner([slot["payload"] for slot in slots])
        except Exception as exc:
            for slot in slots:
                slot["error"] = exc
                slot["event"].set()
            return
        for slot, result in zip(slots, results, strict=True):
            slot["result"] = result
            slot["event"].set()

_batcher = _Batcher(_batch_window_seconds) if _batch_window_seconds > 0 else None

def _generate_codes(
    model: Qwen3TTSModel,
    task: str,
    *,
    text: str,
    voice: str,
    language: str,
    instructions: str,
    max_new_tokens: int,
):
    generate_kwargs = {**QWEN_TTS_GENERATE_KWARGS, "max_new_tokens": max_new_tokens}
    if task == "CustomVoice":
        speaker = voice or DEFAULT_CUSTOMVOICE_SPEAKER
        codes = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=_customvoice_language(speaker, language),
            return_codes_only=True,
            **generate_kwargs,
        )
    elif task == "VoiceDesign":
        codes = model.generate_voice_design(
            text=text,
            language=language,
            instruct=_voice_design_instruction(instructions, voice),
            return_codes_only=True,
            **generate_kwargs,
        )
    else:
        raise ValueError(
            f"_generate_codes does not handle task_type={task!r} "
            "(use generate_voice_clone path)"
        )
    return codes[0] if isinstance(codes, (list, tuple)) else codes

def _to_float32_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32)

def _generate_full_wav(
    model: Qwen3TTSModel, task: str, req: SpeechRequest,
) -> tuple[np.ndarray, int]:
    generate_kwargs = {**QWEN_TTS_GENERATE_KWARGS, "max_new_tokens": req.max_new_tokens}

    if task == "CustomVoice":
        speaker = req.voice or DEFAULT_CUSTOMVOICE_SPEAKER
        wavs, sample_rate = model.generate_custom_voice(
            text=req.input,
            speaker=speaker,
            language=_customvoice_language(speaker, req.language),
            **generate_kwargs,
        )
    elif task == "VoiceDesign":
        wavs, sample_rate = model.generate_voice_design(
            text=req.input,
            language=req.language,
            instruct=_voice_design_instruction(req.instructions, req.voice),
            **generate_kwargs,
        )
    elif task == "Base":
        wavs, sample_rate = _generate_base_wav(model, req, generate_kwargs)
    else:
        raise HTTPException(400, f"Unsupported task_type: {task}")

    first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    return _to_float32_numpy(first), int(sample_rate)

def _spool_ref_audio(spec: str) -> str:
    audio_bytes, suffix = _decode_ref_audio(spec)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        return handle.name

def _generate_base_wav(model: Qwen3TTSModel, req: SpeechRequest, generate_kwargs: dict):
    if req.voice_profile is not None:
        with _voice_profiles_lock:
            profile = _voice_profiles.get(req.voice_profile)
        if profile is None:
            raise HTTPException(404, f"voice_profile {req.voice_profile!r} not found")
        bound_model_id = profile.get("model_id")
        if bound_model_id is not None and _models.get(bound_model_id) is not model:
            raise HTTPException(
                409,
                f"voice_profile {req.voice_profile!r} bound to {bound_model_id!r}, not loaded",
            )
        return model.generate_voice_clone(
            text=req.input,
            language=req.language,
            voice_clone_prompt=profile["prompt"],
            **generate_kwargs,
        )

    if not req.ref_audio:
        raise HTTPException(422, "task_type=Base requires ref_audio or voice_profile")
    return model.generate_voice_clone(
        text=req.input,
        language=req.language,
        ref_audio=_spool_ref_audio(req.ref_audio),
        ref_text=req.ref_text or None,
        x_vector_only_mode=req.x_vector_only_mode,
        **generate_kwargs,
    )

def _task_routing_for_health() -> dict[str, Any]:
    return {
        task: (
            model.config_name_or_path
            if hasattr(model, "config_name_or_path")
            else getattr(model, "model_path", "?")
        )
        for task, model in _models_by_task.items()
    }

def _kokoro_health_block() -> dict[str, Any]:
    voices = _kokoro.voices_list() if _kokoro is not None else []
    return {
        "configured": KOKORO_ENABLED,
        "loaded": _kokoro is not None,
        "load_error": _kokoro_load_error,
        "voices": voices,
    }

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": _device,
        "dtype": DTYPE_NAME,
        "loaded_models": list(_models),
        "task_routing": _task_routing_for_health(),
        "voice_profiles": sorted(_voice_profiles),
        "batch_window_ms": BATCH_WINDOW_MS,
        "stt_backend": {
            "requested": STT_BACKEND_RAW,
            "active": _stt_backend_kind or STT_BACKEND_QWEN3_OMNI,
            "model": _stt_backend_model_id,
            "load_error": _stt_backend_load_error,
        },
        "omni": {
            "configured": bool(OMNI_MODEL_ID),
            "model": OMNI_MODEL_ID or None,
            "loaded": _omni is not None,
            "load_error": _omni_load_error,
            "talker_disabled": OMNI_DISABLE_TALKER,
        },
        "aligner": {
            "configured": bool(ALIGNER_MODEL_ID),
            "model": ALIGNER_MODEL_ID or None,
            "loaded": _aligner is not None,
            "load_error": _aligner_load_error,
            "supported_languages": list(ALIGNER_LANGUAGES),
        },
        "gemma": {
            "configured": bool(GEMMA_MODEL_ID),
            "model": GEMMA_MODEL_ID or None,
            "loaded": _gemma is not None,
            "load_error": _gemma_load_error,
        },
        "kokoro": _kokoro_health_block(),
        "diarizer": {
            "configured": DIAR_ENABLED,
            "segmentation_model": DIAR_SEGMENTATION_MODEL_FILE,
            "embedding_model": DIAR_EMBEDDING_MODEL_FILE,
            "loaded": _diarizer is not None,
            "load_error": _diarizer_load_error,
        },
    }

def _build_models() -> list[oapi.Model]:
    out: list[oapi.Model] = []
    for model_id in _models:
        out.append(oapi.Model(
            id=model_id,
            owned_by=oapi.hf_owner(model_id),
            task=task.TTS,
            extras={"sample_rate": QWEN_TTS_OUTPUT_SR},
        ))
    if OMNI_MODEL_ID:
        out.append(oapi.Model(
            id=OMNI_MODEL_ID, owned_by=oapi.hf_owner(OMNI_MODEL_ID), task=task.CHAT,
        ))
        out.append(oapi.Model(
            id=OMNI_MODEL_ID, owned_by=oapi.hf_owner(OMNI_MODEL_ID), task=task.ASR,
        ))
    if GEMMA_MODEL_ID:
        out.append(oapi.Model(
            id=GEMMA_MODEL_ID, owned_by=oapi.hf_owner(GEMMA_MODEL_ID), task=task.CHAT,
        ))
        out.append(oapi.Model(
            id=GEMMA_MODEL_ID, owned_by=oapi.hf_owner(GEMMA_MODEL_ID), task=task.ASR,
        ))
    if ALIGNER_MODEL_ID:
        out.append(oapi.Model(
            id=ALIGNER_MODEL_ID,
            owned_by=oapi.hf_owner(ALIGNER_MODEL_ID),
            task=task.FORCED_ALIGNMENT,
            languages=[lang.lower() for lang in ALIGNER_LANGUAGES],
        ))
    if _kokoro is not None:
        out.append(oapi.Model(
            id=KOKORO_HF_REPO,
            owned_by=oapi.hf_owner(KOKORO_HF_REPO),
            task=task.TTS,
            languages=list(KOKORO_LANGUAGES),
            extras={"sample_rate": KOKORO_SAMPLE_RATE, "voices": _kokoro.voices_list()},
        ))
    if _stt_backend is not None and _stt_backend_model_id:
        out.append(oapi.Model(
            id=_stt_backend_model_id,
            owned_by=oapi.hf_owner(_stt_backend_model_id),
            task=task.ASR,
        ))
    pii_model_id = os.environ.get("REDACT_MODEL_ID", "").strip()
    if pii_model_id:
        out.append(oapi.Model(
            id=pii_model_id,
            owned_by=oapi.hf_owner(pii_model_id),
            task=task.TOKEN_CLASSIFICATION,
        ))
    return out

@app.get("/v1/models")
def list_models(task: str | None = None) -> dict[str, Any]:
    return oapi.list_models_response(_build_models(), task)

@app.get("/v1/models/{model_id:path}")
def retrieve_model(model_id: str) -> dict[str, Any]:
    for model in _build_models():
        if model.id == model_id:
            return model.to_dict()
    oapi.raise_openai_error(
        404,
        f"model {model_id!r} not found",
        kind.NOT_FOUND,
        param="model",
        code="model_not_found",
    )
    return {}

def _hf_cache_has(model_id: str) -> bool:
    """Return True if `model_id` is already present in HF_HUB_CACHE.

    Used as a cheap idempotency check for `POST /v1/models/{id}` so we can
    no-op instead of trying a runtime download (we run with HF_HUB_OFFLINE=1)."""
    cache_dir = os.environ.get("HF_HUB_CACHE", "").strip()
    if not cache_dir:
        return False
    org_repo = model_id.replace("/", "--")
    return Path(cache_dir, f"models--{org_repo}").exists()

@app.post("/v1/models/{model_id:path}")
def download_model(model_id: str) -> dict[str, Any]:
    """speaches-compat: 'download' is a no-op when the model is already in
    `HF_HUB_CACHE` (our usual case under Nix). For unknown models we return
    503 with a clear message -- runtime downloads aren't supported because the
    cache is managed by `nix/models.nix`."""
    if _hf_cache_has(model_id):
        return {"id": model_id, "status": "already_cached"}
    oapi.raise_openai_error(
        503,
        f"runtime model download not supported; add {model_id!r} to nix/models.nix "
        "and rebuild hf-cache (this server runs with HF_HUB_OFFLINE=1).",
        kind.SERVICE_UNAVAIL, param="model_id", code="model_not_loaded",
    )
    return {}

@app.delete("/v1/models/{model_id:path}")
def delete_model(model_id: str) -> dict[str, Any]:
    """speaches-compat: model cache lifecycle is managed by Nix; we can't
    actually delete a `/nix/store` entry from a running server. Return 405
    so the surface check sees the route as present without claiming we
    performed the destructive op."""
    oapi.raise_openai_error(
        405,
        "runtime model delete not supported; HF cache is managed by Nix",
        kind.INVALID_REQUEST, param="model_id", code="method_not_allowed",
    )
    return {}

@app.get("/v1/audio/voices")
def list_voices() -> dict[str, Any]:
    qwen_voices = [
        {"id": name, "name": name, "language": language, "engine": "qwen-tts"}
        for name, language in CUSTOMVOICE_SPEAKER_LANGUAGES.items()
    ]
    if _kokoro is None:
        return {"data": qwen_voices, "object": "list"}
    kokoro_voices = [
        {"id": name, "name": name, "language": "kokoro", "engine": "kokoro"}
        for name in _kokoro.voices_list()
    ]
    return {"data": qwen_voices + kokoro_voices, "object": "list"}

_AUDIO_TASKS = frozenset({task.TTS, task.ASR, task.FORCED_ALIGNMENT})

@app.get("/v1/audio/models")
def list_audio_models() -> dict[str, Any]:
    """speaches-compat: list models whose task is audio-related."""
    return {"models": [m.to_dict() for m in _build_models() if m.task in _AUDIO_TASKS]}

@app.get("/v1/registry")
def list_registry() -> dict[str, Any]:
    """speaches-compat: list known models (we use the local registry)."""
    return oapi.list_models_response(_build_models(), None)

def _loaded_model_ids() -> list[str]:
    out: list[str] = list(_models.keys())
    if OMNI_MODEL_ID and _omni is not None:
        out.append(OMNI_MODEL_ID)
    if GEMMA_MODEL_ID and _gemma is not None:
        out.append(GEMMA_MODEL_ID)
    if ALIGNER_MODEL_ID and _aligner is not None:
        out.append(ALIGNER_MODEL_ID)
    if _kokoro is not None:
        out.append(KOKORO_HF_REPO)
    if _stt_backend is not None and _stt_backend_model_id:
        out.append(_stt_backend_model_id)
    return out

@app.get("/api/ps")
def list_loaded_models() -> dict[str, Any]:
    """Ollama-compat: flat list of loaded model IDs."""
    return {"models": _loaded_model_ids()}

@app.post("/api/ps/{model_id:path}")
def load_model(model_id: str) -> dict[str, Any]:
    """Ollama-compat: load a model into memory (lazy planes only)."""
    if model_id in _loaded_model_ids():
        oapi.raise_openai_error(
            409, f"Model {model_id!r} is already loaded.",
            kind.INVALID_REQUEST, param="model_id", code="model_already_loaded",
        )
    if OMNI_MODEL_ID and model_id == OMNI_MODEL_ID:
        _load_omni()
        return {"loaded": model_id}
    if GEMMA_MODEL_ID and model_id == GEMMA_MODEL_ID:
        _load_gemma()
        return {"loaded": model_id}
    oapi.raise_openai_error(
        404, f"Model {model_id!r} is not configured; nothing to load.",
        kind.NOT_FOUND, param="model_id", code="model_not_found",
    )
    return {}

@app.delete("/api/ps/{model_id:path}", status_code=204)
def unload_model(model_id: str) -> Response:
    """Ollama-compat: best-effort unload. Most planes are torch graphs that
    can't be safely released mid-process; we honor the call for the lazy
    planes (Omni / Gemma) which we can drop, and no-op otherwise."""
    global _omni, _gemma
    if OMNI_MODEL_ID and model_id == OMNI_MODEL_ID:
        _omni = None
    elif GEMMA_MODEL_ID and model_id == GEMMA_MODEL_ID:
        _gemma = None
    return Response(status_code=204)

@app.post("/v1/audio/speech/timestamps")
async def detect_speech_timestamps(
    file: UploadFile = File(...),
    model: str = Form("silero_vad_v6"),
    threshold: float = Form(0.5),
    neg_threshold: float | None = Form(None),
    min_speech_duration_ms: int = Form(250),
    max_speech_duration_s: float = Form(float("inf")),
    min_silence_duration_ms: int = Form(2000),
    speech_pad_ms: int = Form(400),
) -> dict[str, Any]:
    """speaches-compat: run Silero VAD over the upload and return speech segments."""
    if _realtime_vad_model is None:
        oapi.raise_openai_error(
            503,
            "VAD model not loaded; set VAD_MODEL_FILE to a Silero ONNX path.",
            kind.SERVICE_UNAVAIL, param="model", code="model_not_loaded",
        )
    contents, suffix = await _read_upload(file)
    try:
        from audio.decode_any import decode_any_to_mono_16k
    except ImportError:
        decode_any_to_mono_16k = None  # type: ignore[assignment]
    if decode_any_to_mono_16k is not None:
        audio = decode_any_to_mono_16k(contents, suffix)
    else:
        import io as _io

        audio, _sr = sf.read(_io.BytesIO(contents), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    from vad.constants import WINDOW_SAMPLES
    from vad.segmenter import VadOptions, speech_timestamps_from_probs, to_ms_speech_timestamps

    probs: list[float] = []
    pos = 0
    while pos + WINDOW_SAMPLES <= audio.shape[0]:
        probs.append(_realtime_vad_model.process_window(audio[pos:pos + WINDOW_SAMPLES]))
        pos += WINDOW_SAMPLES
    _realtime_vad_model.reset()
    _max_speech_s = float(max_speech_duration_s)
    if _max_speech_s == float("inf") or _max_speech_s > 86_400:
        _max_speech_s = 86_400.0
    opts = VadOptions(
        threshold=float(threshold),
        neg_threshold=float(neg_threshold) if neg_threshold is not None else max(
            float(threshold) - 0.15, 0.15,
        ),
        min_speech_duration_ms=int(min_speech_duration_ms),
        max_speech_duration_s=_max_speech_s,
        min_silence_duration_ms=int(min_silence_duration_ms),
        speech_pad_ms=int(speech_pad_ms),
    )
    spans = speech_timestamps_from_probs(probs, audio.shape[0], opts, 16_000)
    return {
        "timestamps": [
            {"start": float(s.start) / 1000.0, "end": float(s.end) / 1000.0}
            for s in to_ms_speech_timestamps(spans)
        ],
        "model": model,
    }

@app.post("/v1/audio/speech/embedding")
async def speech_embedding_alias(
    file: UploadFile = File(...),
    model: str = Form("wespeaker-resnet293-LM"),
):
    """speaches-compat: alias for /v1/audio/embeddings using the same backend."""
    return await audio_embeddings(file=[file], audio=[], model=model)

@app.post("/v1/audio/voice-clone")
async def voice_clone(
    file: UploadFile = File(None),
    reference_audio: UploadFile = File(...),
    input: str = Form(...),
    model: str = Form("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    reference_text: str = Form(""),
    mode: str = Form("icl"),
    response_format: Literal["wav", "pcm", "mp3", "flac", "opus", "aac"] = Form("wav"),
    sample_rate: int | None = Form(None),
    language: str = Form(ENGLISH_LANGUAGE),
) -> Response:
    """speaches-compat: synthesize `input` in the voice of `reference_audio`.

    Wraps `/v1/audio/speech` with `task_type="Base"`. `mode` is accepted but
    currently always maps to in-context cloning (we don't implement classifier-
    free guidance or other modes)."""
    ref_bytes, ref_suffix = await _read_upload(reference_audio)
    req = SpeechRequest(
        input=input,
        voice="cloned",
        model=model,
        response_format=response_format,
        sample_rate=sample_rate,
        task_type="Base",
        ref_audio=_audio_data_uri(ref_bytes, ref_suffix),
        ref_text=reference_text,
        language=language,
        x_vector_only_mode=False,
    )
    return synthesize(req)

@app.post("/v1/voice-profiles")
def create_voice_profile(req: VoiceProfileRequest) -> dict[str, Any]:
    if not req.name or "/" in req.name:
        raise HTTPException(422, "voice profile name must be non-empty and slash-free")
    base_model_id = req.model_id or next(
        (
            mid for mid, model in _models.items()
            if _resolve_task_type(mid, None) == "Base"
        ),
        None,
    )
    if base_model_id is None or base_model_id not in _models:
        raise HTTPException(400, "no Base model loaded; can't precompute speaker prompt")
    base_model = _models[base_model_id]
    ref_audio_path = _spool_ref_audio(req.ref_audio)

    try:
        prompt_items = base_model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=req.ref_text or None,
            x_vector_only_mode=req.x_vector_only_mode,
        )
    except ValueError as validation_error:
        raise HTTPException(422, str(validation_error)) from validation_error

    with _voice_profiles_lock:
        _voice_profiles[req.name] = {
            "prompt": prompt_items,
            "model_id": base_model_id,
        }
    return {
        "name": req.name,
        "model_id": base_model_id,
        "x_vector_only_mode": req.x_vector_only_mode,
    }

@app.get("/v1/voice-profiles")
def list_voice_profiles() -> dict[str, Any]:
    with _voice_profiles_lock:
        return {
            "data": [
                {"name": name, "model_id": profile["model_id"]}
                for name, profile in _voice_profiles.items()
            ],
            "object": "list",
        }

@app.delete("/v1/voice-profiles/{name}")
def delete_voice_profile(name: str) -> dict[str, Any]:
    with _voice_profiles_lock:
        if name not in _voice_profiles:
            raise HTTPException(404, f"voice_profile {name!r} not found")
        del _voice_profiles[name]
    return {"deleted": name}

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_CODE_BLOCK_RE = re.compile(r"```tool_code\s*\n(.*?)\n```", re.DOTALL)

def _python_call_to_dict(node: ast.Call, tool_names: set[str]) -> dict[str, Any] | None:
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        return None
    if name not in tool_names:
        return None
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return None
    if node.args:
        positional: list[Any] = []
        for positional_arg in node.args:
            try:
                positional.append(ast.literal_eval(positional_arg))
            except (ValueError, SyntaxError):
                return None
        arguments.setdefault("_positional", positional)
    return _coerce_tool_call_dict({"name": name, "arguments": arguments}, tool_names)

def _walk_tool_code_calls(code: str, tool_names: set[str]) -> list[dict[str, Any]]:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return []
    calls: list[dict[str, Any]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            call = _python_call_to_dict(node, tool_names)
            if call is not None:
                calls.append(call)
    return calls

def _coerce_tool_call_dict(obj: Any, tool_names: set[str]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments")
    if name is None and isinstance(obj.get("function"), dict):
        function_block = obj["function"]
        name = function_block.get("name")
        arguments = function_block.get("arguments", arguments)
    if not isinstance(name, str) or name not in tool_names:
        return None
    if arguments is None:
        arguments_str = "{}"
    elif isinstance(arguments, str):
        try:
            json.loads(arguments)
            arguments_str = arguments
        except (ValueError, TypeError):
            arguments_str = json.dumps({"_raw": arguments})
    else:
        try:
            arguments_str = json.dumps(arguments)
        except (TypeError, ValueError):
            return None
    return {
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": name, "arguments": arguments_str},
    }

def _scan_raw_json_objects(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    found: list[tuple[int, int, Any]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        brace = text.find("{", cursor)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except ValueError:
            cursor = brace + 1
            continue
        found.append((brace, end, obj))
        cursor = end
    return found

def _parse_tool_calls(text: str, tool_names: set[str]) -> list[dict[str, Any]] | None:
    if not text or not tool_names:
        return None

    calls: list[dict[str, Any]] = []
    matched_spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_TAG_RE.finditer(text):
        payload = match.group(1).strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        call = _coerce_tool_call_dict(obj, tool_names)
        if call is not None:
            calls.append(call)
            matched_spans.append(match.span())

    for match in _TOOL_CODE_BLOCK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in matched_spans):
            continue
        for call in _walk_tool_code_calls(match.group(1), tool_names):
            calls.append(call)
            matched_spans.append(match.span())

    for match in _MARKDOWN_JSON_RE.finditer(text):
        if any(start <= match.start() < end for start, end in matched_spans):
            continue
        payload = match.group(1).strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        call = _coerce_tool_call_dict(obj, tool_names)
        if call is not None:
            calls.append(call)
            matched_spans.append(match.span())

    for start, end, obj in _scan_raw_json_objects(text):
        if any(span_start <= start < span_end for span_start, span_end in matched_spans):
            continue
        call = _coerce_tool_call_dict(obj, tool_names)
        if call is not None:
            calls.append(call)
            matched_spans.append((start, end))

    if not calls:
        return None

    calls_with_spans = list(zip(matched_spans, calls, strict=True))
    calls_with_spans.sort(key=lambda pair: pair[0][0])
    return [call for _span, call in calls_with_spans]

def _strip_tool_calls_from_text(text: str, calls: list[dict[str, Any]]) -> str:
    if not text or not calls:
        return text.strip() if text else ""

    tool_names = {call["function"]["name"] for call in calls}
    spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_TAG_RE.finditer(text):
        payload = match.group(1).strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if _coerce_tool_call_dict(obj, tool_names) is not None:
            spans.append(match.span())

    for match in _TOOL_CODE_BLOCK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        if _walk_tool_code_calls(match.group(1), tool_names):
            spans.append(match.span())

    for match in _MARKDOWN_JSON_RE.finditer(text):
        payload = match.group(1).strip()
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if _coerce_tool_call_dict(obj, tool_names) is not None:
            spans.append(match.span())

    for start, end, obj in _scan_raw_json_objects(text):
        if any(span_start <= start < span_end for span_start, span_end in spans):
            continue
        if _coerce_tool_call_dict(obj, tool_names) is not None:
            spans.append((start, end))

    if not spans:
        return text.strip()

    spans.sort()
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            pieces.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces).strip()

def _tool_system_prompt(tools: list[ChatTool], is_gemma: bool) -> str:
    descriptors = []
    for tool in tools:
        function = tool.function
        entry = {
            "name": function.name,
            "description": function.description or "",
            "parameters": function.parameters or {"type": "object", "properties": {}},
        }
        descriptors.append(entry)
    tools_json = json.dumps(descriptors, ensure_ascii=False, indent=2)
    if is_gemma:
        return (
            "You have access to the following tools. When you decide to call a "
            "tool, emit a fenced ```tool_code``` block containing one Python "
            "function call per tool, e.g.:\n"
            "```tool_code\n"
            'get_weather(city="San Francisco", units="celsius")\n'
            "```\n"
            "Use keyword arguments only; values must be Python literals (str, "
            "int, float, bool, None, list, dict). To call multiple tools, put "
            "each call on its own line inside the same block. If no tool is "
            "appropriate, answer normally without a tool_code block.\n\n"
            f"Tools:\n{tools_json}"
        )
    return (
        "You have access to the following tools. When you decide to call a "
        "tool, wrap each call in <tool_call>...</tool_call> tags containing a "
        'JSON object {"name": "<tool_name>", "arguments": {<args>}}. You may '
        "emit multiple <tool_call> blocks. If no tool is appropriate, answer "
        "normally without any <tool_call> tags.\n\n"
        f"Tools:\n{tools_json}"
    )

def _normalize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": [{"type": "text", "text": content}]}
    if content is None:
        return {"role": role, "content": []}
    return {"role": role, "content": list(content)}

def _wants_audio_output(req: ChatCompletionRequest) -> bool:
    return "audio" in req.modalities or req.audio is not None

def _build_chat_response(
    req: ChatCompletionRequest,
    model_id: str,
    message: dict[str, Any],
    usage: dict[str, int],
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        existing = message.get("content")
        if (isinstance(existing, str) and existing.strip() == "") or existing == "":
            message["content"] = None
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }

def _usage_dict(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

def _resolve_tool_calls(
    req: ChatCompletionRequest, text: str,
) -> tuple[list[dict[str, Any]] | None, str]:
    if not req.tools or req.tool_choice == "none":
        return None, text

    tool_names = {tool.function.name for tool in req.tools}
    calls = _parse_tool_calls(text, tool_names)

    if calls and isinstance(req.tool_choice, ChatToolChoiceFunction):
        wanted = req.tool_choice.function.name
        calls = [call for call in calls if call["function"]["name"] == wanted] or None

    if not calls:
        if req.tool_choice == "required":
            raise HTTPException(
                422,
                "tool_choice='required' but the model did not emit a parseable "
                "tool call. Phase 5B-driven grammar enforcement is the proper "
                "fix; for now, retry with stronger system prompt.",
            )
        return None, text

    stripped = _strip_tool_calls_from_text(text, calls)
    return calls, stripped

def _guided_json_from_request(req: ChatCompletionRequest) -> dict[str, Any] | str | None:
    if req.response_format is None or req.response_format.type == "text":
        return None
    if req.response_format.type == "json_object":
        return {}
    if req.response_format.type == "json_schema":
        spec = req.response_format.json_schema
        if spec is None or not spec.schema:
            raise HTTPException(
                400,
                "response_format.type='json_schema' requires a non-empty json_schema.schema dict.",
            )
        return spec.schema
    return None

def _gemma_chat_response(req: ChatCompletionRequest, conversation: list[dict[str, Any]]) -> dict[str, Any]:
    if _wants_audio_output(req):
        raise HTTPException(
            400,
            "Audio output is not supported by Gemma 4 -- only Qwen3-Omni has a talker. "
            "Drop modalities=['audio'] or omit the model field to route to Omni.",
        )
    result = _load_gemma().chat(
        conversation,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        guided_json=_guided_json_from_request(req),
    )
    tool_calls, content_text = _resolve_tool_calls(req, result.text)
    message: dict[str, Any] = {"role": "assistant", "content": content_text}
    return _build_chat_response(
        req,
        GEMMA_MODEL_ID,
        message,
        _usage_dict(result.prompt_tokens, result.completion_tokens),
        tool_calls=tool_calls,
    )

def _omni_audio_message_attachment(audio: np.ndarray, sample_rate: int, transcript: str) -> dict[str, Any]:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "id": f"audio-{uuid.uuid4().hex}",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "format": "wav",
        "transcript": transcript,
    }

def _omni_chat_response(req: ChatCompletionRequest, conversation: list[dict[str, Any]]) -> dict[str, Any]:
    want_audio = _wants_audio_output(req)
    if want_audio and OMNI_DISABLE_TALKER:
        raise HTTPException(
            400, "Audio output requested but server started with QWEN3_OMNI_DISABLE_TALKER=1.",
        )
    speaker = req.audio.voice if req.audio else OMNI_DEFAULT_SPEAKER
    if speaker not in OMNI_SPEAKERS:
        raise HTTPException(
            400, f"voice must be one of {OMNI_SPEAKERS}, got {speaker!r}",
        )

    try:
        result = _load_omni().chat(
            conversation,
            return_audio=want_audio,
            speaker=speaker,
            thinker_max_new_tokens=req.max_tokens,
            talker_max_new_tokens=req.talker_max_new_tokens,
            thinker_temperature=req.temperature,
            use_audio_in_video=req.use_audio_in_video,
            guided_json=_guided_json_from_request(req),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            501,
            "guided decoding via the Omni chat wrapper is not yet supported; "
            "use the Gemma 4 path or wait for Phase 5B-2.",
        ) from exc

    tool_calls, content_text = _resolve_tool_calls(req, result.text)
    message: dict[str, Any] = {"role": "assistant", "content": content_text}
    if want_audio and result.audio is not None:
        message["audio"] = _omni_audio_message_attachment(
            result.audio, result.sample_rate, result.text,
        )

    return _build_chat_response(
        req,
        OMNI_MODEL_ID,
        message,
        _usage_dict(result.prompt_tokens, result.completion_tokens),
        tool_calls=tool_calls,
    )

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if req.stream:
        raise HTTPException(501, "stream=true not yet implemented for /v1/chat/completions")

    is_gemma = _request_picks_gemma(req.model)
    raw_messages = list(req.messages)
    if req.tools and req.tool_choice != "none":
        tool_prompt = _tool_system_prompt(req.tools, is_gemma)
        raw_messages = _inject_tool_system_prompt(raw_messages, tool_prompt)
    conversation = [_normalize_chat_message(message) for message in raw_messages]
    if is_gemma:
        return _gemma_chat_response(req, conversation)
    return _omni_chat_response(req, conversation)

def _inject_tool_system_prompt(
    messages: list[dict[str, Any]], tool_prompt: str,
) -> list[dict[str, Any]]:
    has_system = any(message.get("role") == "system" for message in messages)
    if not has_system:
        return [{"role": "system", "content": tool_prompt}, *messages]
    augmented: list[dict[str, Any]] = []
    appended = False
    for message in messages:
        if not appended and message.get("role") == "system":
            existing = message.get("content")
            if isinstance(existing, str):
                merged = f"{existing}\n\n{tool_prompt}" if existing else tool_prompt
                augmented.append({**message, "content": merged})
            elif isinstance(existing, list):
                augmented.append(
                    {**message, "content": [*existing, {"type": "text", "text": tool_prompt}]},
                )
            else:
                augmented.append({**message, "content": tool_prompt})
            appended = True
        else:
            augmented.append(message)
    return augmented

async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    contents = await file.read()
    if len(contents) > MAX_AUDIO_UPLOAD_BYTES:
        max_mib = MAX_AUDIO_UPLOAD_BYTES // BYTES_PER_MIB
        oapi.raise_openai_error(
            413,
            f"audio upload exceeds {max_mib} MiB",
            kind.INVALID_REQUEST,
            param="file",
            code="payload_too_large",
        )
    suffix = Path(file.filename or "audio").suffix or ".wav"
    return contents, suffix

def _spool_to_tempfile(contents: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(contents)
        return handle.name

_AUDIO_SUFFIX_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".webm": "audio/webm",
}

def _audio_data_uri(contents: bytes, suffix: str) -> str:
    """Encode an uploaded audio blob as a data: URI.

    The multimodal pipeline (audio.loaders.read_bytes_or_b64) rejects absolute
    paths by design -- that validator is for user-supplied specs. Internal
    upload handlers route trusted bytes through this helper instead.
    """
    mime = _AUDIO_SUFFIX_MIME.get(suffix.lower(), "audio/wav")
    return f"data:{mime};base64,{base64.b64encode(contents).decode('ascii')}"

SENTENCE_FRAGMENT_RE = re.compile(r"[^.!?。！？]+[.!?。！？]?", re.UNICODE)

def _segment_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    fragments = [match.group(0).strip() for match in SENTENCE_FRAGMENT_RE.finditer(text)]
    nonempty = [fragment for fragment in fragments if fragment]
    return nonempty or [text]

def _format_timestamp(seconds: float, *, vtt: bool) -> str:
    bounded = max(seconds, 0.0)
    hours_part, remainder = divmod(bounded, SECONDS_PER_HOUR)
    minutes_part, seconds_part = divmod(remainder, SECONDS_PER_MINUTE)
    milliseconds_part = int(round((seconds_part - int(seconds_part)) * MS_PER_SECOND))
    fractional_separator = "." if vtt else ","
    return (
        f"{int(hours_part):02d}:"
        f"{int(minutes_part):02d}:"
        f"{int(seconds_part):02d}"
        f"{fractional_separator}{milliseconds_part:03d}"
    )

def _detect_language_via_omni(audio_path: str) -> str:
    answer = _load_omni().transcribe(
        audio_path,
        prompt=LANGUAGE_DETECTION_PROMPT_TEMPLATE.format(
            options=", ".join(ALIGNER_LANGUAGES),
        ),
    ).strip().lower().rstrip(".")
    for token in re.findall(r"[a-z]+", answer):
        if token in ALIGNER_LANGUAGE_LOOKUP:
            return ALIGNER_LANGUAGE_LOOKUP[token]
    return ENGLISH_LANGUAGE

def _normalize_aligner_language(language: str, audio_path: str) -> str:
    key = (language or "").strip().lower()
    if key in ("", LANGUAGE_AUTO):
        return _detect_language_via_omni(audio_path)
    if key in ALIGNER_LANGUAGE_LOOKUP:
        return ALIGNER_LANGUAGE_LOOKUP[key]
    raise HTTPException(
        422,
        f"language={language!r} not supported by the forced aligner. "
        f"Supported: {sorted(ALIGNER_LANGUAGES)} or 'auto'.",
    )

def _alphanumeric_lowercase(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())

def _bucket_aligner_items_into_sentences(
    items: list, sentences: list[str],
) -> tuple[list[float], list[float]]:
    cue_starts: list[float] = []
    cue_ends: list[float] = []
    sentence_targets = [_alphanumeric_lowercase(sentence) for sentence in sentences]
    item_cursor = 0
    for target in sentence_targets:
        if item_cursor >= len(items):
            cue_starts.append(items[-1].end_time)
            cue_ends.append(items[-1].end_time)
            continue
        cue_start = items[item_cursor].start_time
        accumulated = ""
        cue_end = items[item_cursor].end_time
        while item_cursor < len(items) and len(accumulated) < len(target):
            accumulated += _alphanumeric_lowercase(items[item_cursor].text)
            cue_end = items[item_cursor].end_time
            item_cursor += 1
        cue_starts.append(cue_start)
        cue_ends.append(cue_end)
    return cue_starts, cue_ends

def _empty_caption_block(text: str, *, vtt: bool) -> str:
    header = "WEBVTT\n\n" if vtt else ""
    zero = _format_timestamp(0, vtt=vtt)
    return f"{header}{zero} --> {zero}\n{text}\n"

def _format_caption_blocks(
    sentences: list[str],
    cue_starts: list[float],
    cue_ends: list[float],
    *,
    vtt: bool,
) -> str:
    blocks: list[str] = ["WEBVTT", ""] if vtt else []
    for cue_index, sentence in enumerate(sentences):
        if not vtt:
            blocks.append(str(cue_index + 1))
        start_ts = _format_timestamp(cue_starts[cue_index], vtt=vtt)
        end_ts = _format_timestamp(cue_ends[cue_index], vtt=vtt)
        blocks.append(f"{start_ts} --> {end_ts}")
        blocks.append(sentence)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"

def _load_audio_for_diarizer(audio_path: str) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if sample_rate != 16_000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
    return waveform.astype(np.float32, copy=False)

def _aligner_items_for(
    audio_path: str, text: str, language_field: str,
) -> tuple[list, str | None]:
    if _aligner is None or not text.strip():
        return [], None
    try:
        normalized = _normalize_aligner_language(language_field, audio_path)
    except HTTPException:
        return [], None
    try:
        [result] = _aligner.align(audio=audio_path, text=text, language=normalized)
    except Exception:
        return [], normalized
    return list(result), normalized

def _assign_aligner_items_to_diar(
    diar: list[DiarSegment], items: list,
) -> list[list]:
    buckets: list[list] = [[] for _ in diar]
    if not diar:
        return buckets
    for item in items:
        mid_ms = (item.start_time + item.end_time) * 500.0
        chosen = 0
        best_dist = float("inf")
        for i, seg in enumerate(diar):
            if seg.t_start_ms <= mid_ms <= seg.t_end_ms:
                chosen = i
                break
            dist = (
                seg.t_start_ms - mid_ms if mid_ms < seg.t_start_ms
                else mid_ms - seg.t_end_ms
            )
            if dist < best_dist:
                best_dist = dist
                chosen = i
        buckets[chosen].append(item)
    return buckets

def _diarized_segments_for(
    audio_path: str, text: str, language_field: str,
) -> list[dict[str, Any]]:
    if _diarizer is None:
        return []
    try:
        audio = _load_audio_for_diarizer(audio_path)
        diar = _diarizer.diarize_utterance(audio, t_start_ms=0)
    except Exception:
        return []
    aligner_items, _normalized = _aligner_items_for(audio_path, text, language_field)
    buckets = _assign_aligner_items_to_diar(diar, aligner_items)
    out: list[dict[str, Any]] = []
    for idx, (seg, items) in enumerate(zip(diar, buckets, strict=True), start=1):
        seg_text = " ".join(item.text for item in items if item.text).strip()
        out.append({
            "type": "transcript.text.segment",
            "id": f"seg_{idx:03d}",
            "speaker": f"SPEAKER_{seg.speaker:02d}",
            "start": round(seg.t_start_ms / 1000.0, 3),
            "end": round(seg.t_end_ms / 1000.0, 3),
            "duration": round(max(seg.t_end_ms - seg.t_start_ms, 0) / 1000.0, 3),
            "text": seg_text,
            "avg_logprob": None,
            "no_speech_prob": None,
            "confidence": float(seg.confidence),
        })
    return out

def _captions(
    audio_path: str, text: str, language: str, *, fmt: Literal["srt", "vtt"],
) -> str:
    if _aligner is None:
        reason = (
            f"Load failed at boot: {_aligner_load_error}"
            if _aligner_load_error
            else "Server started with QWEN3_ALIGNER_MODEL=''."
        )
        raise HTTPException(503, f"SRT/VTT requires the forced aligner. {reason}")

    is_vtt = fmt == "vtt"
    text = text.strip()
    sentences = _segment_text(text)
    if not text or not sentences:
        return _empty_caption_block("", vtt=is_vtt)

    [result] = _aligner.align(audio=audio_path, text=text, language=language)
    items = list(result)
    if not items:
        return _empty_caption_block(text, vtt=is_vtt)

    cue_starts, cue_ends = _bucket_aligner_items_into_sentences(items, sentences)
    return _format_caption_blocks(sentences, cue_starts, cue_ends, vtt=is_vtt)

def _transcribe_with_picked_engine(
    model_field: str, audio_spec: str, *, language: str | None, prompt: str | None,
) -> str:
    engine = _load_gemma() if _request_picks_gemma(model_field) else _load_omni()
    return engine.transcribe(audio_spec, language=language, prompt=prompt or None)

def _request_picks_qwen3_omni_stt(model_field: str | None) -> bool:
    if not model_field:
        return False
    if not OMNI_MODEL_ID:
        return False
    if model_field == OMNI_MODEL_ID:
        return True
    return _model_basename(model_field).startswith("qwen3-omni")

def _request_picks_whisper_stt(model_field: str | None) -> bool:
    if _stt_backend is None:
        return False
    if _request_picks_qwen3_omni_stt(model_field):
        return False
    return True

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("default"),
    language: str = Form("auto"),
    prompt: str = Form(""),
    response_format: Literal[
        "json", "text", "verbose_json", "srt", "vtt", "diarized_json",
    ] = Form("json"),
):
    if _request_picks_whisper_stt(model):
        from stt import http as stt_http
        await file.seek(0)
        normalized_language = None if language in (None, "", "auto") else language
        normalized_prompt = prompt or None
        return await stt_http.transcriptions_post(
            backend=_stt_backend,
            file=file,
            response_format=response_format,
            language=normalized_language,
            prompt=normalized_prompt,
        )
    contents, suffix = await _read_upload(file)
    tmp_path = _spool_to_tempfile(contents, suffix)
    audio_spec = _audio_data_uri(contents, suffix)
    detected_language: str | None = None
    captions_body: str | None = None
    diarized_segments: list[dict[str, Any]] | None = None
    try:
        try:
            text = _transcribe_with_picked_engine(
                model, audio_spec, language=language, prompt=prompt,
            )
        except HTTPException:
            raise
        except Exception as exc:
            oapi.raise_openai_error(
                400,
                f"audio decode: {exc}",
                kind.INVALID_REQUEST,
                param="file",
                code="audio_decode_error",
            )
        if response_format in ("srt", "vtt"):
            detected_language = _normalize_aligner_language(language, tmp_path)
            captions_body = _captions(
                tmp_path, text, detected_language, fmt=response_format,
            )
        if response_format == "diarized_json":
            diarized_segments = _diarized_segments_for(tmp_path, text, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if response_format == "text":
        return Response(text, media_type=PLAIN_TEXT_MEDIA_TYPE)
    if response_format == "srt":
        return Response(captions_body, media_type=SRT_MEDIA_TYPE)
    if response_format == "vtt":
        return Response(captions_body, media_type=VTT_MEDIA_TYPE)
    if response_format == "diarized_json":
        return JSONResponse({
            "text": text,
            "avg_logprob": None,
            "no_speech_prob": None,
            "segments": diarized_segments or [{
                "type": "transcript.text.segment",
                "id": "seg_001",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 0.0,
                "duration": 0.0,
                "text": text,
                "avg_logprob": None,
                "no_speech_prob": None,
                "confidence": None,
            }],
        })
    if response_format == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": detected_language or language,
            "duration": _audio_duration_seconds_or_none(contents),
            "text": text,
            "segments": [],
            "words": [],
        })
    payload: dict[str, Any] = {"text": text, "language": language, "model": model}
    if detected_language is not None:
        payload["detected_language"] = detected_language
    return JSONResponse(payload)

@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form("default"),
    prompt: str = Form(""),
    response_format: Literal["json", "text"] = Form("json"),
):
    if _request_picks_whisper_stt(model):
        from stt import http as stt_http
        await file.seek(0)
        return await stt_http.translations_post(
            backend=_stt_backend,
            file=file,
            response_format=response_format,
            prompt=prompt or None,
        )
    contents, suffix = await _read_upload(file)
    audio_spec = _audio_data_uri(contents, suffix)
    translation_prompt = prompt or DEFAULT_TRANSLATION_PROMPT
    engine = _load_gemma() if _request_picks_gemma(model) else _load_omni()
    try:
        text = engine.transcribe(audio_spec, prompt=translation_prompt)
    except HTTPException:
        raise
    except Exception as exc:
        oapi.raise_openai_error(
            400,
            f"audio decode: {exc}",
            kind.INVALID_REQUEST,
            param="file",
            code="audio_decode_error",
        )

    if response_format == "text":
        return Response(text, media_type=PLAIN_TEXT_MEDIA_TYPE)
    return JSONResponse({"text": text, "model": model})

DEFAULT_DIAR_FILE_ID = "audio"
DEFAULT_EMBEDDING_MODEL_NAME = "wespeaker-resnet293-LM"
RTTM_MEDIA_TYPE = "text/plain; charset=utf-8"

def _decode_data_url_with_mime(spec: str) -> tuple[bytes, str | None]:
    s = spec.strip()
    if not s.startswith("data:"):
        raise ValueError("not a data URL")
    rest = s[len("data:"):]
    comma = rest.find(",")
    if comma < 0:
        raise ValueError("missing comma")
    header = rest[:comma]
    body = rest[comma + 1:]
    mime: str | None = None
    is_b64 = False
    for i, part in enumerate(header.split(";")):
        if i == 0 and part:
            mime = part
            continue
        if part.lower() == "base64":
            is_b64 = True
    if not is_b64:
        raise ValueError("only base64 data URLs are supported")
    try:
        decoded = base64.b64decode(body, validate=True)
    except Exception as exc:
        raise ValueError(f"base64: {exc}") from exc
    return decoded, mime

def _decode_uploaded_audio_to_16k(
    contents: bytes, mime: str | None, suffix_hint: str | None = None,
) -> np.ndarray:
    suffix = suffix_hint or _suffix_from_mime(mime) or ".wav"
    tmp_path = _spool_to_tempfile(contents, suffix)
    try:
        return _load_audio_for_diarizer(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

_MIME_SUFFIX_MAP = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
}

def _suffix_from_mime(mime: str | None) -> str | None:
    if not mime:
        return None
    return _MIME_SUFFIX_MAP.get(mime.split(";", 1)[0].strip().lower())

def _file_id_from_filename(filename: str | None) -> str:
    if not filename:
        return DEFAULT_DIAR_FILE_ID
    stem = Path(filename).stem
    return stem or DEFAULT_DIAR_FILE_ID

def _build_speaker_label_map(
    segments: list[DiarSegment],
    audio: np.ndarray,
    emb: EmbeddingModel,
    known: list[tuple[str, np.ndarray]],
):
    per_cluster: dict[int, list[np.ndarray]] = {}
    for s in segments:
        start_idx = (int(s.t_start_ms) * 16_000) // 1000
        end_idx = (int(s.t_end_ms) * 16_000) // 1000
        end_idx = min(end_idx, audio.shape[0])
        if end_idx <= start_idx:
            continue
        per_cluster.setdefault(int(s.speaker), []).append(audio[start_idx:end_idx])

    cluster_to_known: dict[int, str] = {}
    if known:
        for cid, slices in per_cluster.items():
            pooled = np.concatenate(slices) if len(slices) > 1 else slices[0]
            if pooled.shape[0] < emb.min_input_samples:
                continue
            try:
                cluster_emb = emb.embed(pooled)
            except Exception:
                continue
            best_name: str | None = None
            best_sim = -2.0
            for name, kv in known:
                sim = cosine_sim(cluster_emb, kv)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            if best_name is not None:
                cluster_to_known[cid] = best_name

    def label_for(cid: int) -> str:
        if cid in cluster_to_known:
            return cluster_to_known[cid]
        return f"SPEAKER_{cid:02d}"

    return label_for

@app.post("/v1/audio/diarization")
async def diarization(
    file: UploadFile = File(...),
    model: str = Form(""),
    response_format: str = Form("json"),
    known_speaker_names: list[str] = Form(default=[], alias="known_speaker_names[]"),
    known_speaker_references: list[str] = Form(
        default=[], alias="known_speaker_references[]",
    ),
):
    if _diarizer is None or _diar_embedding is None:
        oapi.raise_openai_error(
            503,
            "diarization model not loaded; run scripts/fetch-models.sh and scripts/export-diarizen-onnx.py",
            kind.SERVICE_UNAVAIL,
            code="model_not_loaded",
        )
    contents, suffix = await _read_upload(file)
    mime = file.content_type
    try:
        samples = _decode_uploaded_audio_to_16k(contents, mime, suffix)
    except Exception as exc:
        oapi.raise_openai_error(
            400,
            f"audio decode: {exc}",
            kind.INVALID_REQUEST,
            param="file",
            code="audio_decode_error",
        )
    duration_s = float(samples.shape[0]) / 16_000.0

    emb = _diar_embedding
    known_embeddings: list[tuple[str, np.ndarray]] = []
    if known_speaker_names and len(known_speaker_names) == len(known_speaker_references):
        for name, data_url in zip(
            known_speaker_names, known_speaker_references, strict=True,
        ):
            try:
                ref_bytes, ref_mime = _decode_data_url_with_mime(data_url)
            except Exception as exc:
                oapi.raise_openai_error(
                    400,
                    f"known_speaker_references[{name}]: {exc}",
                    kind.INVALID_REQUEST,
                    param="known_speaker_references",
                    code="data_url_decode_error",
                )
            try:
                ref_samples = _decode_uploaded_audio_to_16k(ref_bytes, ref_mime)
            except Exception as exc:
                oapi.raise_openai_error(
                    400,
                    f"known_speaker_references[{name}] decode: {exc}",
                    kind.INVALID_REQUEST,
                    param="known_speaker_references",
                    code="audio_decode_error",
                )
            try:
                vec = emb.embed(ref_samples)
            except Exception as exc:
                oapi.raise_openai_error(
                    500,
                    f"embed reference {name}: {exc}",
                    kind.SERVER,
                    code="embed_failed",
                )
            known_embeddings.append((name, vec))

    diarizer = _diarizer
    with _diarizer_lock:
        diarizer.reset()
        try:
            segments = diarizer.diarize_utterance(samples, t_start_ms=0)
        except Exception as exc:
            oapi.raise_openai_error(
                500,
                f"diarize: {exc}",
                kind.SERVER,
                code="diarize_failed",
            )

    label_for = _build_speaker_label_map(segments, samples, emb, known_embeddings)

    if response_format == "rttm":
        file_id = _file_id_from_filename(file.filename)
        lines: list[str] = []
        for s in segments:
            start = s.t_start_ms / 1000.0
            dur = max(s.t_end_ms - s.t_start_ms, 0) / 1000.0
            lines.append(
                f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> "
                f"{label_for(int(s.speaker))} <NA> <NA>\n",
            )
        return Response("".join(lines), media_type=RTTM_MEDIA_TYPE)

    return JSONResponse({
        "duration": duration_s,
        "segments": [
            {
                "start": s.t_start_ms / 1000.0,
                "end": s.t_end_ms / 1000.0,
                "speaker": label_for(int(s.speaker)),
            }
            for s in segments
        ],
    })

@app.post("/v1/audio/embeddings")
async def audio_embeddings(
    file: list[UploadFile] = File(default=[]),
    audio: list[str] = Form(default=[]),
    model: str = Form(""),
):
    if _diar_embedding is None:
        oapi.raise_openai_error(
            503,
            "embedding model not loaded; run scripts/fetch-models.sh",
            kind.SERVICE_UNAVAIL,
            code="model_not_loaded",
        )
    inputs: list[tuple[bytes, str | None, str | None]] = []
    for upload in file:
        contents, suffix = await _read_upload(upload)
        inputs.append((contents, upload.content_type, suffix))
    for data_url in audio:
        try:
            ref_bytes, ref_mime = _decode_data_url_with_mime(data_url)
        except Exception as exc:
            oapi.raise_openai_error(
                400,
                f"audio data URL: {exc}",
                kind.INVALID_REQUEST,
                param="audio",
                code="data_url_decode_error",
            )
        inputs.append((ref_bytes, ref_mime, None))

    if not inputs:
        oapi.raise_fastapi_validation_error([oapi.missing_field(["body", "file"])])

    emb = _diar_embedding
    data_items: list[dict[str, Any]] = []
    total_seconds = 0.0
    for idx, (contents, mime, suffix) in enumerate(inputs):
        try:
            samples = _decode_uploaded_audio_to_16k(contents, mime, suffix)
        except Exception as exc:
            oapi.raise_openai_error(
                400,
                f"audio decode (file index {idx}): {exc}",
                kind.INVALID_REQUEST,
                param="file",
                code="audio_decode_error",
            )
        if samples.shape[0] < emb.min_input_samples:
            oapi.raise_openai_error(
                400,
                f"input audio too short (file index {idx}, {samples.shape[0]} samples; "
                f"need >={emb.min_input_samples})",
                kind.INVALID_REQUEST,
                param="file",
                code="audio_too_short",
            )
        total_seconds += float(samples.shape[0]) / 16_000.0
        try:
            vector = emb.embed(samples)
        except Exception as exc:
            oapi.raise_openai_error(
                500,
                f"embed (file index {idx}): {exc}",
                kind.SERVER,
                code="embed_failed",
            )
        data_items.append({
            "object": "embedding",
            "index": idx,
            "embedding": vector.tolist(),
        })

    return JSONResponse({
        "object": "list",
        "data": data_items,
        "model": model or DEFAULT_EMBEDDING_MODEL_NAME,
        "usage": {"audio_seconds": total_seconds},
    })

def _kokoro_language_from_request(req: SpeechRequest) -> str:
    if req.language and req.language.lower() != LANGUAGE_AUTO:
        return req.language
    return KOKORO_DEFAULT_LANGUAGE

def _kokoro_has_voice(kokoro: KokoroTTS, name: str) -> bool:
    try:
        return name in kokoro.voices_list()
    except (OSError, ValueError):
        return False

def _resolve_kokoro_voice(kokoro: KokoroTTS, requested: str) -> str:
    if not requested:
        return KOKORO_DEFAULT_VOICE
    if not _kokoro_has_voice(kokoro, requested) and is_openai_voice_alias(requested):
        return KOKORO_DEFAULT_VOICE
    return requested

def _streaming_response_for_format(
    pcm_iter,
    source_sr: int,
    target_format: str,
    target_sr: int | None,
):
    effective_target_sr = target_sr or source_sr
    if target_format == "pcm":
        return StreamingResponse(
            pcm_iter,
            media_type=SPEECH_FORMAT_MIME["pcm"],
        )
    _require_ffmpeg(target_format)
    encoded = _pcm_chunks_through_ffmpeg(pcm_iter, source_sr, effective_target_sr, target_format)
    return StreamingResponse(encoded, media_type=SPEECH_FORMAT_MIME[target_format])

def _sse_response_from_pcm(pcm_iter) -> StreamingResponse:
    return StreamingResponse(
        _sse_speech_events(pcm_iter),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache"},
    )

def _oneshot_response_for_format(
    audio: np.ndarray,
    source_sr: int,
    target_format: str,
    target_sr: int | None,
):
    effective_target_sr = target_sr or source_sr
    if target_format == "pcm":
        body = f32_to_s16le(audio)
        return Response(body, media_type=SPEECH_FORMAT_MIME["pcm"])
    if target_format == "wav" and target_sr in (None, source_sr):
        return _wav_response(audio, source_sr)
    _require_ffmpeg(target_format)
    pcm_bytes = f32_to_s16le(audio)
    encoded = _encode_pcm_once(
        pcm_bytes, source_sr, effective_target_sr, target_format,
    )
    return Response(encoded, media_type=SPEECH_FORMAT_MIME[target_format])

def _kokoro_synthesize_response(req: SpeechRequest):
    kokoro = _kokoro_or_503()
    voice = _resolve_kokoro_voice(kokoro, req.voice)
    language = _kokoro_language_from_request(req)
    stream_mode = _resolve_stream_format(req)

    if stream_mode is not None:
        pcm_iter = _kokoro_pcm_stream(kokoro, req.input, voice, req.speed, language)
        if stream_mode == "sse":
            return _sse_response_from_pcm(pcm_iter)
        return _streaming_response_for_format(
            pcm_iter, KOKORO_SAMPLE_RATE, req.response_format, req.sample_rate,
        )

    audio, sample_rate = kokoro.synthesize(
        req.input, voice, speed=req.speed, lang=language,
    )
    return _oneshot_response_for_format(
        audio, sample_rate, req.response_format, req.sample_rate,
    )

def _qwen_tts_synthesize_response(req: SpeechRequest):
    if not _models:
        oapi.raise_openai_error(
            503, "no models loaded", kind.SERVICE_UNAVAIL,
        )

    task = req.task_type or _resolve_task_type(MODEL_IDS[0], None)
    model = _model_for_task(task)
    stream_mode = _resolve_stream_format(req)

    if stream_mode is not None:
        if task == "Base":
            oapi.raise_openai_error(
                400,
                "streaming is not supported for task_type=Base",
                kind.INVALID_REQUEST,
                param="task_type",
                code="stream_unsupported",
            )
        pcm_iter = _stream_pcm_via_thread(model, task, req)
        if stream_mode == "sse":
            return _sse_response_from_pcm(pcm_iter)
        return _streaming_response_for_format(
            pcm_iter,
            QWEN_TTS_OUTPUT_SR,
            req.response_format,
            req.sample_rate,
        )

    if _batcher is not None and task in ("CustomVoice", "VoiceDesign"):
        audio, sample_rate = _batcher.submit(
            key=(id(model), task),
            payload={"req": req, "task": task},
            runner=lambda payloads: _run_batch(model, task, payloads),
        )
        return _oneshot_response_for_format(
            audio, sample_rate, req.response_format, req.sample_rate,
        )

    audio, sample_rate = _generate_full_wav(model, task, req)
    return _oneshot_response_for_format(
        audio, sample_rate, req.response_format, req.sample_rate,
    )

class _RealtimeModelsView:
    @property
    def model_ids(self) -> list[str]:
        ids = list(_models.keys())
        if OMNI_MODEL_ID and OMNI_MODEL_ID not in ids:
            ids.append(OMNI_MODEL_ID)
        if GEMMA_MODEL_ID and GEMMA_MODEL_ID not in ids:
            ids.append(GEMMA_MODEL_ID)
        return ids

    @property
    def diarizer(self):
        return _diarizer

    @property
    def kokoro(self):
        return _kokoro

_realtime_models_view = _RealtimeModelsView()

@app.get("/v1/realtime/capabilities")
def _realtime_capabilities():
    from realtime import capabilities_json_with_models
    return JSONResponse(capabilities_json_with_models(_realtime_models_view))

@app.get("/health/sessions")
def _realtime_sessions_health():
    from realtime.transport import webrtc_session_count
    from realtime.websocket import active_session_count
    ws_count = int(active_session_count())
    rtc_count = int(webrtc_session_count())
    return JSONResponse(
        {
            "live_sessions": ws_count + rtc_count,
            "ws_sessions": ws_count,
            "webrtc_sessions": rtc_count,
        }
    )

_SDP_CLIENT_ERROR_NEEDLES = (
    "parse offer SDP",
    "set_remote_description",
    "syntax error",
    "no ice-ufrag",
    "no ice-pwd",
    "no fingerprint",
    "unable to start",
    "SdpInvalidSyntax",
    "SdpEmpty",
    "Invalid SDP",
    "SDP has no",
)

def _is_client_sdp_error(msg: str) -> bool:
    if not msg:
        return False
    return any(needle in msg for needle in _SDP_CLIENT_ERROR_NEEDLES)

@app.post("/v1/realtime")
async def _realtime_offer(request: Request):
    from realtime import RealtimeQuery
    from realtime.transport import maybe_handle_offer

    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    body_bytes = await request.body()
    if content_type == "application/sdp":
        offer_sdp = body_bytes.decode("utf-8", errors="replace")
    elif content_type == "application/json":
        try:
            obj = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            oapi.raise_openai_error(
                400, "invalid JSON body", kind.INVALID_REQUEST,
                code="invalid_json",
            )
        sdp = obj.get("sdp") if isinstance(obj, dict) else None
        if not isinstance(sdp, str) or not sdp:
            oapi.raise_openai_error(
                400, "expected JSON {\"sdp\":...,\"type\":\"offer\"}",
                kind.INVALID_REQUEST, param="sdp", code="invalid_request",
            )
        offer_sdp = sdp
    else:
        oapi.raise_openai_error(
            415, "expected Content-Type: application/sdp or application/json",
            kind.INVALID_REQUEST, code="unsupported_media_type",
        )

    params = dict(request.query_params)
    query = RealtimeQuery(
        intent=params.get("intent"),
        voice=params.get("voice"),
        model=params.get("model"),
        transcription_model=params.get("transcription_model"),
        language=params.get("language"),
    )

    try:
        answer_sdp = await maybe_handle_offer(offer_sdp, query)
    except Exception as exc:
        msg = str(exc)
        if _is_client_sdp_error(msg):
            oapi.raise_openai_error(
                400, msg, kind.INVALID_REQUEST, code="sdp_invalid",
            )
        oapi.raise_openai_error(
            500, f"realtime negotiate failed: {exc}", kind.SERVER,
            code="negotiate_failed",
        )

    if answer_sdp is None:
        oapi.raise_openai_error(
            503, "WebRTC support unavailable (aiortc not installed)",
            kind.SERVICE_UNAVAIL, code="webrtc_unavailable",
        )

    return PlainTextResponse(answer_sdp, media_type="application/sdp")

@app.websocket("/v1/realtime")
async def _realtime_ws(websocket: WebSocket):
    from realtime.websocket import realtime_ws_endpoint
    await realtime_ws_endpoint(websocket)

@app.post("/v1/audio/speech")
def synthesize(req: SpeechRequest):
    _validate_speech_request(req)
    req.input = _clean_speech_input(req.input)
    if not req.input:
        return Response(b"", media_type=SPEECH_FORMAT_MIME[req.response_format])
    if _request_picks_kokoro(req):
        return _kokoro_synthesize_response(req)
    return _qwen_tts_synthesize_response(req)

def _kokoro_pcm_stream(
    kokoro: KokoroTTS, text: str, voice: str, speed: float, language: str,
):
    for audio, _sample_rate in kokoro.stream(text, voice, speed=speed, lang=language):
        yield f32_to_s16le(audio)

def _stream_pcm_via_thread(model: Qwen3TTSModel, task: str, req: SpeechRequest):
    completion_event = threading.Event()
    shared: dict[str, Any] = {}

    def runner():
        try:
            shared["codes"] = _generate_codes(
                model, task,
                text=req.input, voice=req.voice, language=req.language,
                instructions=req.instructions, max_new_tokens=req.max_new_tokens,
            )
        except Exception as exc:
            shared["error"] = exc
        finally:
            completion_event.set()

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()

    def chunked_pcm_iterator():
        completion_event.wait()
        if "error" in shared:
            raise shared["error"]
        yield from chunked_decode_pcm(model.model, shared["codes"])

    return chunked_pcm_iterator()

def _customvoice_batch_inputs(payloads: list[dict]) -> tuple[list[str], list[str], list[str]]:
    speakers = [
        payload["req"].voice or DEFAULT_CUSTOMVOICE_SPEAKER for payload in payloads
    ]
    languages = [
        _customvoice_language(speaker, payload["req"].language)
        for payload, speaker in zip(payloads, speakers, strict=True)
    ]
    texts = [payload["req"].input for payload in payloads]
    return texts, speakers, languages

def _voice_design_batch_inputs(payloads: list[dict]) -> tuple[list[str], list[str], list[str]]:
    instructs = [
        _voice_design_instruction(payload["req"].instructions, payload["req"].voice)
        for payload in payloads
    ]
    languages = [payload["req"].language for payload in payloads]
    texts = [payload["req"].input for payload in payloads]
    return texts, languages, instructs

def _run_batch(
    model: Qwen3TTSModel, task: str, payloads: list[dict],
) -> list[tuple[np.ndarray, int]]:
    max_new_tokens = max(payload["req"].max_new_tokens for payload in payloads)
    generate_kwargs = {**QWEN_TTS_GENERATE_KWARGS, "max_new_tokens": max_new_tokens}

    if task == "CustomVoice":
        texts, speakers, languages = _customvoice_batch_inputs(payloads)
        wavs, sample_rate = model.generate_custom_voice(
            text=texts, speaker=speakers, language=languages, **generate_kwargs,
        )
    else:
        texts, languages, instructs = _voice_design_batch_inputs(payloads)
        wavs, sample_rate = model.generate_voice_design(
            text=texts, language=languages, instruct=instructs, **generate_kwargs,
        )

    return [(_to_float32_numpy(wav), int(sample_rate)) for wav in wavs]

def _wav_response(audio: np.ndarray, sample_rate: int) -> StreamingResponse:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type=WAV_MEDIA_TYPE)

_pii_classifier: Any = None
_pii_classifier_lock = threading.Lock()
_pii_load_error: str | None = None

def _load_pii_classifier():
    global _pii_classifier, _pii_load_error
    if _pii_classifier is not None:
        return _pii_classifier
    model_id = os.environ.get("REDACT_MODEL_ID", "").strip()
    if not model_id:
        raise HTTPException(503, "PII classifier disabled. Set REDACT_MODEL_ID to enable.")
    with _pii_classifier_lock:
        if _pii_classifier is not None:
            return _pii_classifier
        if _pii_load_error is not None:
            raise HTTPException(503, f"PII classifier load previously failed: {_pii_load_error}")
        try:
            from pii.classifier import PiiClassifier
            _pii_classifier = PiiClassifier()
        except Exception as exc:
            _pii_load_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(503, f"PII classifier failed to load: {_pii_load_error}") from exc
    return _pii_classifier

class PiiClassifyIn(BaseModel):
    text: str

class PiiClassifyBatchIn(BaseModel):
    texts: list[str]

class _PiiSpanOut(BaseModel):
    start: int
    endExclusive: int
    label: str

@app.post("/v1/pii/classify")
def pii_classify(req: PiiClassifyIn) -> dict:
    clf = _load_pii_classifier()
    spans = clf.classify_one(req.text)
    return {"spans": [{"start": s.start, "endExclusive": s.endExclusive, "label": s.label} for s in spans]}

@app.post("/v1/pii/classify/batch")
def pii_classify_batch(req: PiiClassifyBatchIn) -> dict:
    from pii.classifier import MAX_BATCH
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(413, f"batch size {len(req.texts)} exceeds {MAX_BATCH}")
    clf = _load_pii_classifier()
    batched = clf.classify_batch(req.texts)
    return {
        "results": [
            {"spans": [{"start": s.start, "endExclusive": s.endExclusive, "label": s.label} for s in spans]}
            for spans in batched
        ]
    }

@app.post("/v1/pii/redact/analyze")
async def pii_redact_analyze(file: UploadFile = File(...)):
    try:
        from pii.ocr import run_ocr
    except ImportError:
        raise HTTPException(status_code=503, detail="OCR backend not available")
    from pii.span_mapper import map_spans

    contents = await file.read()
    try:
        from PIL import Image as _PILImage
        pil_img = _PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_array = np.array(pil_img)
    image_width, image_height = pil_img.size

    try:
        ocr_result = run_ocr(img_array)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    clf = _load_pii_classifier()
    spans = clf.classify_one(ocr_result.text)
    labeled_rects = map_spans(ocr_result.tokens, spans, image_width, image_height)

    return JSONResponse({
        "text": ocr_result.text,
        "tokens": [
            {
                "start": t.start,
                "endExclusive": t.end_exclusive,
                "rect": {"left": t.left, "top": t.top, "right": t.right, "bottom": t.bottom},
            }
            for t in ocr_result.tokens
        ],
        "spans": [
            {"start": s.start, "endExclusive": s.endExclusive, "label": s.label}
            for s in spans
        ],
        "rects": [
            {"left": r.left, "top": r.top, "right": r.right, "bottom": r.bottom, "label": r.label}
            for r in labeled_rects
        ],
    })

@app.post("/v1/pii/redact/render")
async def pii_redact_render(
    file: UploadFile = File(...),
    rects: str = Form(...),
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
):
    from pii.renderer import Rect, render

    contents = await file.read()
    try:
        from PIL import Image as _PILImage
        pil_img = _PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        rects_data = json.loads(rects)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid rects JSON")

    if not isinstance(rects_data, list):
        raise HTTPException(status_code=400, detail="rects must be a JSON array")

    parsed_rects: list[Rect] = []
    for r in rects_data:
        try:
            parsed_rects.append(Rect(
                left=int(r["left"]),
                top=int(r["top"]),
                right=int(r["right"]),
                bottom=int(r["bottom"]),
            ))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Each rect must have left, top, right, bottom")

    if fill_mode not in ("solid", "shuffle"):
        raise HTTPException(status_code=400, detail="fill_mode must be 'solid' or 'shuffle'")

    png_bytes = render(pil_img, parsed_rects, fill_mode=fill_mode, fill_color=fill_color)
    return Response(content=png_bytes, media_type="image/png")

def main() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
