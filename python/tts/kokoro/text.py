from __future__ import annotations

import re
from enum import Enum

import numpy as np

KOKORO_SAMPLE_RATE = 24_000
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 48_000
SPEED_MIN = 0.5
SPEED_MAX = 2.0
DEFAULT_SPEED = 1.0
MAX_CHUNK_CHARS = 400
DEFAULT_VOICE = "af_heart"
DEFAULT_LANGUAGE = "en-us"

KOKORO_LANGUAGES = ("en", "es", "fr", "hi", "it", "ja", "pt", "zh")
KOKORO_VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "en", "b": "en",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt",
    "z": "zh",
}

PCM_INT16_SCALE = 32767.0
PCM_INT16_MIN = -32768
PCM_INT16_MAX = 32767
PCM_S16LE_DTYPE = "<i2"

OPENAI_VOICE_ALIASES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"},
)

class ResponseFormat(str, Enum):
    PCM = "pcm"
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OPUS = "opus"
    AAC = "aac"

    def mime_type(self) -> str:
        return _RESPONSE_FORMAT_MIME[self]

_RESPONSE_FORMAT_MIME: dict[ResponseFormat, str] = {
    ResponseFormat.PCM: "audio/pcm",
    ResponseFormat.MP3: "audio/mpeg",
    ResponseFormat.WAV: "audio/wav",
    ResponseFormat.FLAC: "audio/flac",
    ResponseFormat.OPUS: "audio/opus",
    ResponseFormat.AAC: "audio/aac",
}

class StreamFormat(str, Enum):
    AUDIO = "audio"
    SSE = "sse"

def is_openai_voice_alias(name: str) -> bool:
    return name.lower() in OPENAI_VOICE_ALIASES

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "✂-➰"
    "]+",
)
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC_STAR_RE = re.compile(r"\*(.*?)\*")
_MD_UNDER_RE = re.compile(r"__(.*?)__")
_MD_ITALIC_UNDER_RE = re.compile(r"_(.*?)_")
_WHITESPACE_RE = re.compile(r"\s+")
_NEWLINE_RE = re.compile(r"[\r\n]+")

def strip_emojis(s: str) -> str:
    return _EMOJI_RE.sub("", s)

def strip_markdown_emphasis(s: str) -> str:
    out = _MD_BOLD_RE.sub(r"\1", s)
    out = _MD_ITALIC_STAR_RE.sub(r"\1", out)
    out = _MD_UNDER_RE.sub(r"\1", out)
    return _MD_ITALIC_UNDER_RE.sub(r"\1", out)

def normalize_for_tts(s: str) -> str:
    collapsed_nl = _NEWLINE_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", collapsed_nl).strip()

def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
            end = i + 1
            k = end
            while k < len(text) and text[k].isspace():
                k += 1
            out.append(text[start:end])
            start = k
            i = k
            continue
        i += 1
    if start < len(text):
        out.append(text[start:])
    return out

def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if max_chars == 0:
        max_chars = MAX_CHUNK_CHARS
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        trimmed = current.strip()
        if trimmed:
            chunks.append(trimmed)
        current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            _flush()
            for word in sentence.split():
                projected = len(current) + len(word) + (0 if not current else 1)
                if projected <= max_chars:
                    current = f"{current} {word}".strip() if current else word
                else:
                    _flush()
                    current = word
        else:
            projected = len(current) + len(sentence) + (0 if not current else 1)
            if projected <= max_chars:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                _flush()
                current = sentence
    _flush()
    return chunks

def f32_to_s16le(samples: np.ndarray) -> bytes:
    finite = np.nan_to_num(
        samples.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0,
    )
    scaled = np.round(finite * PCM_INT16_SCALE)
    clamped = np.clip(scaled, PCM_INT16_MIN, PCM_INT16_MAX)
    return clamped.astype(PCM_S16LE_DTYPE).tobytes()
