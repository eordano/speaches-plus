from __future__ import annotations

import numpy as np

ULAW_BIAS: int = 0x84
ULAW_CLIP: int = 32_635

def _leading_seg_8bit(v: int) -> int:
    v &= 0xFF
    if v & 0x80:
        return 7
    if v & 0x40:
        return 6
    if v & 0x20:
        return 5
    if v & 0x10:
        return 4
    if v & 0x08:
        return 3
    if v & 0x04:
        return 2
    if v & 0x02:
        return 1
    return 0

def _to_i16(v: int) -> int:
    v &= 0xFFFF
    if v >= 0x8000:
        v -= 0x10000
    return v

def ulaw_decode_byte(u: int) -> int:
    u = (~u) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    t = (mantissa << 3) + ULAW_BIAS
    t <<= exponent
    s = (ULAW_BIAS - t) if sign != 0 else (t - ULAW_BIAS)
    return _to_i16(s)

def ulaw_encode_sample(s: int) -> int:
    sample = int(s)
    if sample < 0:
        sample = -sample
        sign = 0x80
    else:
        sign = 0
    if sample > ULAW_CLIP:
        sample = ULAW_CLIP
    sample += ULAW_BIAS
    exponent = _leading_seg_8bit((sample >> 7) & 0xFF)
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF

def alaw_decode_byte(a: int) -> int:
    a = (a ^ 0x55) & 0xFF
    mantissa = a & 0x0F
    seg = (a & 0x70) >> 4
    t = mantissa << 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    val = t if (a & 0x80) != 0 else -t
    if val > 32_767:
        val = 32_767
    elif val < -32_768:
        val = -32_768
    return val

_ALAW_SEG_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)

def alaw_encode_sample(s: int) -> int:
    pcm = int(s)
    if pcm >= 0:
        mask = 0xD5
    else:
        pcm = -pcm - 8
        mask = 0x55
    seg = 8
    for i, end in enumerate(_ALAW_SEG_END):
        if pcm <= end:
            seg = i
            break
    if seg >= 8:
        return (0x7F ^ mask) & 0xFF
    aval = (seg & 0xFF) << 4
    if seg < 2:
        mantissa = (pcm >> 4) & 0x0F
    else:
        mantissa = (pcm >> (seg + 3)) & 0x0F
    return ((aval | mantissa) ^ mask) & 0xFF

_ULAW_DECODE_TABLE = np.array(
    [ulaw_decode_byte(i) for i in range(256)], dtype=np.int16
)
_ALAW_DECODE_TABLE = np.array(
    [alaw_decode_byte(i) for i in range(256)], dtype=np.int16
)

_ENCODE_LUT_INDICES = np.arange(65536, dtype=np.uint16).view(np.int16)
_ULAW_ENCODE_LUT = np.array(
    [ulaw_encode_sample(int(x)) for x in _ENCODE_LUT_INDICES], dtype=np.uint8
)
_ALAW_ENCODE_LUT = np.array(
    [alaw_encode_sample(int(x)) for x in _ENCODE_LUT_INDICES], dtype=np.uint8
)

def ulaw_bytes_to_f32(b: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    arr = np.frombuffer(bytes(b), dtype=np.uint8)
    return (_ULAW_DECODE_TABLE[arr].astype(np.float32)) / np.float32(32768.0)

def alaw_bytes_to_f32(b: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    arr = np.frombuffer(bytes(b), dtype=np.uint8)
    return (_ALAW_DECODE_TABLE[arr].astype(np.float32)) / np.float32(32768.0)

def f32_to_ulaw_bytes(samples: np.ndarray) -> bytes:
    s = np.asarray(samples, dtype=np.float32)
    s = np.clip(s, -1.0, 1.0)
    v = np.rint(s * 32767.0).astype(np.int32)
    v = np.clip(v, -32768, 32767).astype(np.int16)
    return _ULAW_ENCODE_LUT[v.view(np.uint16)].tobytes()

def f32_to_alaw_bytes(samples: np.ndarray) -> bytes:
    s = np.asarray(samples, dtype=np.float32)
    s = np.clip(s, -1.0, 1.0)
    v = np.rint(s * 32767.0).astype(np.int32)
    v = np.clip(v, -32768, 32767).astype(np.int16)
    return _ALAW_ENCODE_LUT[v.view(np.uint16)].tobytes()

__all__ = [
    "ULAW_BIAS",
    "ULAW_CLIP",
    "ulaw_decode_byte",
    "ulaw_encode_sample",
    "alaw_decode_byte",
    "alaw_encode_sample",
    "ulaw_bytes_to_f32",
    "alaw_bytes_to_f32",
    "f32_to_ulaw_bytes",
    "f32_to_alaw_bytes",
]
