from __future__ import annotations

import struct

import numpy as np

from .resample import downmix_and_resample_f32
from .types import S16_SCALE, S24_SCALE, S32_SCALE, TARGET_SAMPLE_RATE

def find_chunk(buf: bytes | bytearray | memoryview, tag: bytes) -> int | None:
    if len(tag) != 4:
        raise ValueError("tag must be exactly 4 bytes")
    b = bytes(buf)
    i = 12
    n = len(b)
    while i + 8 <= n:
        chunk_tag = b[i : i + 4]
        size = struct.unpack_from("<I", b, i + 4)[0]
        if chunk_tag == tag:
            return i
        if size == 0xFFFFFFFF:
            return i
        next_i = i + 8 + size + (size & 1)
        if next_i <= i:
            return None
        i = next_i
    return None

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_KSDATAFORMAT_SUBTYPE_PCM = bytes.fromhex("0100000000001000800000aa00389b71")
_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")

def _decode_int_samples(
    data: bytes, channels: int, bits_per_sample: int
) -> np.ndarray:
    if bits_per_sample == 16:
        arr = np.frombuffer(data, dtype="<i2").astype(np.float32)
        return arr / np.float32(S16_SCALE)
    if bits_per_sample == 24:
        n = len(data) // 3
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        raw = np.frombuffer(data[: n * 3], dtype=np.uint8).reshape(n, 3)
        as_i32 = (
            raw[:, 0].astype(np.uint32)
            | (raw[:, 1].astype(np.uint32) << 8)
            | (raw[:, 2].astype(np.uint32) << 16)
        )
        sign = (as_i32 & 0x800000).astype(np.uint32)
        as_signed = as_i32.astype(np.int64)
        as_signed = np.where(sign != 0, as_signed - (1 << 24), as_signed)
        return (as_signed.astype(np.float32)) / np.float32(S24_SCALE)
    if bits_per_sample == 32:
        arr = np.frombuffer(data, dtype="<i4").astype(np.float32)
        return arr / np.float32(S32_SCALE)
    raise ValueError(f"unsupported bits_per_sample: {bits_per_sample}")

def decode_wav_to_16k_mono(data: bytes | bytearray | memoryview) -> np.ndarray:
    if data is None or len(data) < 12:
        raise ValueError("wav: input too short")

    buf = bytearray(data)
    if (
        len(buf) >= 12
        and bytes(buf[0:4]) == b"RIFF"
        and bytes(buf[8:12]) == b"WAVE"
        and bytes(buf[4:8]) == b"\xff\xff\xff\xff"
    ):
        riff_size = len(buf) - 8
        struct.pack_into("<I", buf, 4, riff_size & 0xFFFFFFFF)
        data_idx = find_chunk(bytes(buf), b"data")
        if data_idx is not None and data_idx + 8 <= len(buf):
            if bytes(buf[data_idx + 4 : data_idx + 8]) == b"\xff\xff\xff\xff":
                data_size = len(buf) - data_idx - 8
                struct.pack_into("<I", buf, data_idx + 4, data_size & 0xFFFFFFFF)

    if bytes(buf[0:4]) != b"RIFF" or bytes(buf[8:12]) != b"WAVE":
        raise ValueError("wav: not a RIFF/WAVE file")

    fmt_idx = find_chunk(bytes(buf), b"fmt ")
    if fmt_idx is None:
        raise ValueError("wav: missing fmt chunk")
    fmt_size = struct.unpack_from("<I", buf, fmt_idx + 4)[0]
    if fmt_size < 16:
        raise ValueError(f"wav: fmt chunk too small ({fmt_size})")
    fmt_body = bytes(buf[fmt_idx + 8 : fmt_idx + 8 + fmt_size])
    format_tag = struct.unpack_from("<H", fmt_body, 0)[0]
    channels = struct.unpack_from("<H", fmt_body, 2)[0]
    sample_rate = struct.unpack_from("<I", fmt_body, 4)[0]
    bits_per_sample = struct.unpack_from("<H", fmt_body, 14)[0]

    if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(fmt_body) >= 40:
        subtype = fmt_body[24:40]
        if subtype == _KSDATAFORMAT_SUBTYPE_PCM:
            format_tag = _WAVE_FORMAT_PCM
        elif subtype == _KSDATAFORMAT_SUBTYPE_IEEE_FLOAT:
            format_tag = _WAVE_FORMAT_IEEE_FLOAT

    if channels == 0:
        raise ValueError("wav: zero channels")

    data_idx = find_chunk(bytes(buf), b"data")
    if data_idx is None:
        raise ValueError("wav: missing data chunk")
    data_size = struct.unpack_from("<I", buf, data_idx + 4)[0]
    data_start = data_idx + 8
    if data_size == 0xFFFFFFFF or data_start + data_size > len(buf):
        data_size = len(buf) - data_start
    data_bytes = bytes(buf[data_start : data_start + data_size])

    if format_tag == _WAVE_FORMAT_PCM:
        f = _decode_int_samples(data_bytes, channels, bits_per_sample)
    elif format_tag == _WAVE_FORMAT_IEEE_FLOAT:
        if bits_per_sample == 32:
            f = np.frombuffer(data_bytes, dtype="<f4").astype(np.float32)
        elif bits_per_sample == 64:
            f = np.frombuffer(data_bytes, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(
                f"wav: unsupported float bits_per_sample: {bits_per_sample}"
            )
    else:
        raise ValueError(f"wav: unsupported format tag {format_tag}")

    return downmix_and_resample_f32(
        f, int(channels), int(sample_rate), int(TARGET_SAMPLE_RATE)
    )

def encode_wav_mono16(samples: np.ndarray, sample_rate: int) -> bytes:
    s = np.asarray(samples, dtype=np.float32)
    s = np.clip(s, -1.0, 1.0)
    v = np.rint(s * 32767.0).astype(np.int32)
    v = np.clip(v, -32768, 32767).astype("<i2")
    data = v.tobytes()
    num_channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    data_size = len(data)
    total_size = 36 + data_size
    header = b"RIFF" + struct.pack("<I", total_size) + b"WAVE"
    fmt_chunk = (
        b"fmt "
        + struct.pack("<I", 16)
        + struct.pack(
            "<HHIIHH",
            1,
            num_channels,
            int(sample_rate),
            int(sample_rate) * num_channels * bytes_per_sample,
            num_channels * bytes_per_sample,
            bits_per_sample,
        )
    )
    data_chunk = b"data" + struct.pack("<I", data_size) + data
    return header + fmt_chunk + data_chunk

__all__ = [
    "find_chunk",
    "decode_wav_to_16k_mono",
    "encode_wav_mono16",
]
