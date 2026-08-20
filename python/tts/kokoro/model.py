from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

import env
from .npz import Voice, load_voices
from .phonemize import configure_espeak, phonemize
from .text import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    KOKORO_LANGUAGES,
    KOKORO_SAMPLE_RATE,
    KOKORO_VOICE_PREFIX_TO_LANG,
    SPEED_MAX,
    SPEED_MIN,
)
from .vocab import MAX_PHONEME_LENGTH, PAD_TOKEN_ID, clean_phonemes, tokenize

KOKORO_HF_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILENAME_IN_REPO = "onnx/model.onnx"
KOKORO_VOICE_DIR_IN_REPO = "voices"

DEFAULT_PRELOAD_VOICES = (
    "af_heart", "af_bella", "af_aoede",
    "am_adam", "am_michael",
    "bf_emma", "bm_george",
)

PROVIDER_PRIORITY = (
    "CoreMLExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "WebGpuExecutionProvider",
    "DmlExecutionProvider",
)
PROVIDER_FALLBACK = "CPUExecutionProvider"

NEW_INPUT_TOKEN_KEY = "input_ids"
LEGACY_INPUT_TOKEN_KEY = "tokens"
STYLE_INPUT_KEY = "style"
SPEED_INPUT_KEY = "speed"

VOICE_TENSOR_LAST_DIM = 256
VOICE_FILE_SUFFIX = ".bin"
VOICES_ARCHIVE_SUFFIXES = (".bin", ".npz")

SENTENCE_PUNCTUATION = ".,!?;"
_SENTENCE_SPLIT_RE = re.compile(r"([.,!?;])")

def _ensure_model_file() -> str:
    override = env.read_str_or_none(env.KOKORO_MODEL_FILE)
    if override:
        return override
    from huggingface_hub import hf_hub_download
    return hf_hub_download(KOKORO_HF_REPO, KOKORO_MODEL_FILENAME_IN_REPO)

def _ensure_voice_file(voice_name: str, voices_dir: str | None) -> str:
    if voices_dir:
        local_path = Path(voices_dir) / f"{voice_name}{VOICE_FILE_SUFFIX}"
        if local_path.exists():
            return str(local_path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        KOKORO_HF_REPO,
        f"{KOKORO_VOICE_DIR_IN_REPO}/{voice_name}{VOICE_FILE_SUFFIX}",
    )

def _list_voices_from_hf_repo() -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    return sorted(
        Path(filename).stem
        for filename in api.list_repo_files(KOKORO_HF_REPO)
        if filename.startswith(f"{KOKORO_VOICE_DIR_IN_REPO}/")
        and filename.endswith(VOICE_FILE_SUFFIX)
    )

def _load_voice_tensor(path: str) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.float32)
    return flat.reshape(-1, 1, VOICE_TENSOR_LAST_DIM)

def _voice_to_tensor(voice: Voice) -> np.ndarray:
    arr = np.asarray(voice.data, dtype=np.float32)
    return arr.reshape(*voice.shape) if len(voice.shape) > 1 else arr.reshape(-1, 1, VOICE_TENSOR_LAST_DIM)

def _is_voices_archive(voices_dir: str | None) -> Path | None:
    if not voices_dir:
        return None
    p = Path(voices_dir)
    if p.is_file() and p.suffix.lower() in VOICES_ARCHIVE_SUFFIXES:
        return p
    return None

def _pick_onnx_providers() -> list[str]:
    override = env.read_str_or_none(env.KOKORO_ONNX_PROVIDER)
    if override:
        return [name.strip() for name in override.split(",") if name.strip()]
    available = set(ort.get_available_providers())
    selected = [name for name in PROVIDER_PRIORITY if name in available]
    selected.append(PROVIDER_FALLBACK)
    return selected

def split_phoneme_chunks(phonemes: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(phonemes)
    chunks: list[str] = []
    current = ""
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        too_long = len(current) + len(part) + 1 >= MAX_PHONEME_LENGTH
        if too_long:
            if current.strip():
                chunks.append(current.strip())
            current = part
        elif part in SENTENCE_PUNCTUATION:
            current += part
        else:
            current = (current + " " + part) if current else part
    if current.strip():
        chunks.append(current.strip())
    return chunks

def _validate_speed(speed: float) -> None:
    if not SPEED_MIN <= speed <= SPEED_MAX:
        raise ValueError(f"speed must be in [{SPEED_MIN}, {SPEED_MAX}], got {speed}")

class KokoroTTS:
    def __init__(
        self,
        voices_dir: str | None = None,
        preload_voices: tuple[str, ...] = DEFAULT_PRELOAD_VOICES,
    ):
        configure_espeak()
        self.model_path = _ensure_model_file()
        self.voices_dir = voices_dir

        self.session = ort.InferenceSession(
            self.model_path, providers=_pick_onnx_providers(),
        )
        self._input_names = {spec.name for spec in self.session.get_inputs()}

        self._voice_cache: dict[str, np.ndarray] = {}
        archive = _is_voices_archive(voices_dir)
        self._voices_archive: dict[str, Voice] | None = None
        if archive is not None:
            self._voices_archive = load_voices(archive)
            for voice_name in preload_voices:
                voice = self._voices_archive.get(voice_name)
                if voice is None:
                    warnings.warn(
                        f"Kokoro preload skipped {voice_name!r}: not in archive {archive}",
                        stacklevel=2,
                    )
                    continue
                self._voice_cache[voice_name] = _voice_to_tensor(voice)
        else:
            for voice_name in preload_voices:
                try:
                    self._voice_cache[voice_name] = _load_voice_tensor(
                        _ensure_voice_file(voice_name, voices_dir),
                    )
                except FileNotFoundError as exc:
                    warnings.warn(f"Kokoro preload skipped {voice_name!r}: {exc}", stacklevel=2)
        self._known_voices: list[str] | None = None

    @property
    def voices(self) -> dict[str, np.ndarray]:
        return self._voice_cache

    def has_voice(self, name: str) -> bool:
        return name in self.voices_list()

    def voices_list(self) -> list[str]:
        if self._known_voices is not None:
            return list(self._known_voices)
        if self._voices_archive is not None:
            self._known_voices = sorted(self._voices_archive.keys())
            return list(self._known_voices)
        if self.voices_dir:
            local = Path(self.voices_dir)
            if local.is_dir():
                self._known_voices = sorted(
                    path.stem for path in local.glob(f"*{VOICE_FILE_SUFFIX}")
                )
                return list(self._known_voices)
        try:
            self._known_voices = _list_voices_from_hf_repo()
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Kokoro voice list fetch failed ({exc}); falling back to preloaded cache",
                stacklevel=2,
            )
            self._known_voices = sorted(self._voice_cache.keys())
        return list(self._known_voices)

    def _voice_tensor(self, name: str) -> np.ndarray:
        cached = self._voice_cache.get(name)
        if cached is not None:
            return cached
        if self._voices_archive is not None:
            voice = self._voices_archive.get(name)
            if voice is None:
                raise FileNotFoundError(f"voice {name!r} not in archive")
            tensor = _voice_to_tensor(voice)
            self._voice_cache[name] = tensor
            return tensor
        path = _ensure_voice_file(name, self.voices_dir)
        tensor = _load_voice_tensor(path)
        self._voice_cache[name] = tensor
        return tensor

    def _resolve_voice_style(self, voice: str | np.ndarray) -> np.ndarray:
        if isinstance(voice, str):
            return self._voice_tensor(voice)
        return voice

    def _phonemize_and_clean(self, text: str, language: str) -> str:
        raw = phonemize(text, language)
        cleaned = clean_phonemes(raw)
        if not cleaned:
            raise ValueError(f"phonemize produced empty output for {text!r}")
        return cleaned

    def _run_chunk(
        self, phonemes: str, voice_style: np.ndarray, speed: float,
    ) -> np.ndarray:
        token_ids = tokenize(phonemes)
        if not token_ids:
            raise ValueError(f"tokenize empty for cleaned phonemes {phonemes!r}")
        if len(token_ids) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"phoneme token count {len(token_ids)} exceeds "
                f"MAX_PHONEME_LENGTH ({MAX_PHONEME_LENGTH})"
            )
        style_for_length = voice_style[len(token_ids)]
        token_tensor = np.array(
            [[PAD_TOKEN_ID, *token_ids, PAD_TOKEN_ID]], dtype=np.int64,
        )
        token_input_key = (
            NEW_INPUT_TOKEN_KEY
            if NEW_INPUT_TOKEN_KEY in self._input_names
            else LEGACY_INPUT_TOKEN_KEY
        )
        session_inputs = {
            token_input_key: token_tensor,
            STYLE_INPUT_KEY: np.asarray(style_for_length, dtype=np.float32),
            SPEED_INPUT_KEY: np.array([speed], dtype=np.float32),
        }
        return self.session.run(None, session_inputs)[0].reshape(-1)

    def synthesize(
        self,
        text: str,
        voice: str | np.ndarray = DEFAULT_VOICE,
        *,
        speed: float = DEFAULT_SPEED,
        lang: str = DEFAULT_LANGUAGE,
    ) -> tuple[np.ndarray, int]:
        _validate_speed(speed)
        voice_style = self._resolve_voice_style(voice)
        phonemes = self._phonemize_and_clean(text, lang)
        chunks = split_phoneme_chunks(phonemes) or [phonemes]
        per_chunk_audio = [self._run_chunk(chunk, voice_style, speed) for chunk in chunks]
        return np.concatenate(per_chunk_audio).astype(np.float32), KOKORO_SAMPLE_RATE

    def stream(
        self,
        text: str,
        voice: str | np.ndarray = DEFAULT_VOICE,
        *,
        speed: float = DEFAULT_SPEED,
        lang: str = DEFAULT_LANGUAGE,
    ) -> Iterator[tuple[np.ndarray, int]]:
        _validate_speed(speed)
        voice_style = self._resolve_voice_style(voice)
        phonemes = self._phonemize_and_clean(text, lang)
        for chunk in split_phoneme_chunks(phonemes) or [phonemes]:
            audio = self._run_chunk(chunk, voice_style, speed).astype(np.float32)
            yield audio, KOKORO_SAMPLE_RATE
