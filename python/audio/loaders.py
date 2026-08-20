from __future__ import annotations

import base64
import binascii
import io
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf

DEFAULT_AUDIO_SR = 16000
DEFAULT_VIDEO_MAX_FRAMES = 32
DEFAULT_VIDEO_FALLBACK_FPS = 30.0
RAW_INPUT_FRAME_BUFFER_FACTOR = 8

MULTIMODAL_MAX_BYTES = 64 * 1024 * 1024
MULTIMODAL_FETCH_TIMEOUT_SECONDS = 10
MULTIMODAL_ALLOWED_SCHEMES = ("https",)
BARE_BASE64_MIN_LENGTH = 256
BARE_BASE64_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)
SPEC_REJECT_PREFIXES = ("http://", "file://", "ftp://", "ftps://", "gopher://", "/")

def _enforce_size(payload: bytes, label: str) -> bytes:
    if len(payload) > MULTIMODAL_MAX_BYTES:
        cap_mib = MULTIMODAL_MAX_BYTES // (1024 * 1024)
        raise ValueError(f"{label} exceeds {cap_mib} MiB cap")
    return payload

def _decode_data_uri(spec: str) -> bytes:
    if ";base64," not in spec:
        raise ValueError("data: URI must include ';base64,'; raw text payloads are not accepted")
    payload = spec.split(",", 1)[1]
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"data: URI base64 decode failed: {exc}") from exc
    return _enforce_size(decoded, "data: URI payload")

def _fetch_https(spec: str) -> bytes:
    parsed = urlparse(spec)
    if parsed.scheme not in MULTIMODAL_ALLOWED_SCHEMES:
        raise ValueError(
            f"scheme {parsed.scheme!r} not allowed; multimodal input requires "
            "https:// or data:audio/...;base64,..."
        )
    if not parsed.hostname:
        raise ValueError("https:// URL is missing a hostname")
    with urlopen(spec, timeout=MULTIMODAL_FETCH_TIMEOUT_SECONDS) as response:
        data = response.read(MULTIMODAL_MAX_BYTES + 1)
    return _enforce_size(data, "https:// download")

def _looks_like_bare_base64(spec: str) -> bool:
    if len(spec) < BARE_BASE64_MIN_LENGTH:
        return False
    return all(char in BARE_BASE64_ALPHABET for char in spec)

def _decode_bare_base64(spec: str) -> bytes:
    try:
        decoded = base64.b64decode(spec, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"bare base64 decode failed: {exc}") from exc
    return _enforce_size(decoded, "bare base64 payload")

def read_bytes_or_b64(spec: str) -> bytes:
    if not isinstance(spec, str) or not spec:
        raise ValueError("multimodal input spec must be a non-empty string")
    if "\x00" in spec:
        raise ValueError("multimodal input spec contains a NUL byte")
    if spec.startswith("data:"):
        return _decode_data_uri(spec)
    if spec.startswith("https://"):
        return _fetch_https(spec)
    if spec.startswith(SPEC_REJECT_PREFIXES) or ".." in spec:
        raise ValueError(
            "multimodal input must be data:...;base64,... or https://; "
            "http://, file://, absolute paths, and traversal are blocked"
        )
    if _looks_like_bare_base64(spec):
        return _decode_bare_base64(spec)
    raise ValueError(
        "multimodal input spec is neither a data: URI, https:// URL, nor "
        f"bare base64 (>={BARE_BASE64_MIN_LENGTH} chars, base64 alphabet only)"
    )

def load_audio(spec: str, *, target_sr: int = DEFAULT_AUDIO_SR) -> np.ndarray:
    waveform, sample_rate = sf.read(
        io.BytesIO(read_bytes_or_b64(spec)), dtype="float32", always_2d=False,
    )
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = waveform.astype(np.float32)
    if sample_rate != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    return waveform

def load_image(spec: str):
    from PIL import Image
    return Image.open(io.BytesIO(read_bytes_or_b64(spec))).convert("RGB")

def _uniformly_sampled_indices(num_frames: int, num_samples: int) -> np.ndarray:
    return np.linspace(0, num_frames - 1, num_samples).round().astype(int)

def _strided_sample(frames: list, source_fps: float, target_fps: float, max_frames: int) -> list:
    stride = max(1, int(round(source_fps / target_fps)))
    return frames[::stride][:max_frames]

def _read_video_frames(spec: str, max_frames: int) -> tuple[list, float]:
    import imageio.v3 as iio

    blob = read_bytes_or_b64(spec)
    metadata = iio.immeta(blob, plugin="pyav")
    source_fps = float(metadata.get("fps", 0.0)) or DEFAULT_VIDEO_FALLBACK_FPS
    raw_frames: list = []
    raw_frame_cap = max_frames * RAW_INPUT_FRAME_BUFFER_FACTOR
    for frame in iio.imiter(blob, plugin="pyav"):
        raw_frames.append(frame)
        if len(raw_frames) >= raw_frame_cap:
            break
    return raw_frames, source_fps

def load_video(
    spec: str,
    *,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    target_fps: float | None = None,
) -> list:
    from PIL import Image

    raw_frames, source_fps = _read_video_frames(spec, max_frames)
    if not raw_frames:
        return []

    if target_fps is not None and target_fps > 0:
        sampled = _strided_sample(raw_frames, source_fps, target_fps, max_frames)
    elif len(raw_frames) <= max_frames:
        sampled = raw_frames
    else:
        indices = _uniformly_sampled_indices(len(raw_frames), max_frames)
        sampled = [raw_frames[index] for index in indices]

    return [Image.fromarray(frame).convert("RGB") for frame in sampled]

def _rewrite_part(part: dict) -> dict:
    kind = part.get("type")
    if kind == "input_audio":
        return {"type": "audio", "audio": ""}
    if kind in ("image_url", "video_url"):
        target_kind = kind.removesuffix("_url")
        inner = part.get(kind) or {}
        url = inner.get("url") if isinstance(inner, dict) else inner
        return {"type": target_kind, target_kind: url or ""}
    return part

def normalize_parts(conversation: list[dict]) -> list[dict]:
    rewritten: list[dict] = []
    for message in conversation:
        content = message.get("content")
        role = message.get("role", "user")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        rewritten.append({
            "role": role,
            "content": [_rewrite_part(part) for part in content],
        })
    return rewritten

def _audio_spec_from_part(part: dict) -> str | None:
    kind = part.get("type")
    if kind == "audio":
        return part.get("audio") or part.get("url")
    if kind == "input_audio":
        inner = part.get("input_audio") or {}
        return inner.get("data") or inner.get("url")
    return None

def _image_spec_from_part(part: dict) -> str | None:
    kind = part.get("type")
    if kind == "image":
        return part.get("image") or part.get("url")
    if kind == "image_url":
        inner = part.get("image_url") or {}
        return inner.get("url") if isinstance(inner, dict) else inner
    return None

def _video_spec_from_part(part: dict) -> str | None:
    kind = part.get("type")
    if kind == "video":
        return part.get("video") or part.get("url")
    if kind == "video_url":
        inner = part.get("video_url") or {}
        return inner.get("url") if isinstance(inner, dict) else inner
    return None

def process_mm_info(
    conversation: list[dict],
    *,
    audio_sr: int = DEFAULT_AUDIO_SR,
    video_max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    video_target_fps: float | None = None,
) -> tuple[list, list, list]:
    audios: list[np.ndarray] = []
    images: list[Any] = []
    videos: list[Any] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            audio_spec = _audio_spec_from_part(part)
            if audio_spec is not None:
                audios.append(load_audio(audio_spec, target_sr=audio_sr))
                continue
            image_spec = _image_spec_from_part(part)
            if image_spec is not None:
                images.append(load_image(image_spec))
                continue
            video_spec = _video_spec_from_part(part)
            if video_spec is not None:
                videos.append(load_video(
                    video_spec,
                    max_frames=video_max_frames,
                    target_fps=video_target_fps,
                ))
    return audios, images, videos
